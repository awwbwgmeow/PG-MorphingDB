
#include "model/model_manager.h"
#include "catalog/model_info_d.h"
#ifdef __cplusplus
extern "C" {
#endif

#include "postgres.h"
#include "utils/relcache.h"
#include "utils/syscache.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "storage/lockdefs.h"
#include "access/table.h"
#include "access/htup_details.h"
#include "catalog/model_info.h"
#include "catalog/model_layer_info.h"
#include "catalog/base_model_info.h"
#include "utils/memutils.h"
#include "utils/catcache.h"
#include "utils/vector_tensor.h"

extern char pkglib_path[];

ModelManager model_manager;

bool 
model_manager_load_model(ModelManager *manager, const char *model_path, const char *model_name, const char *base_model)
{
    Relation    pg_model_layer_info_rel; 
    size_t      parm_vector_size;  

    if(manager->module_handle_.find(model_path) != manager->module_handle_.end()){
        return true;
    }

    // load base model
    try {
        torch::jit::script::Module cur_module = torch::jit::load(model_path);
        manager->module_handle_[model_path].first = cur_module;
        manager->module_handle_[model_path].second = at::kCPU;
        manager->module_handle_[model_path].first.to(manager->module_handle_[model_path].second);
        manager->module_handle_[model_path].first.eval();
        //return true;
    }
    catch (const std::exception& e) {
        ereport(ERROR, (errmsg("load model failed, error message: %s", e.what())));
        return false;
    }

    // load parameter
    if(model_name != NULL){
        // read layer parameter in layer_model_info
        pg_model_layer_info_rel = table_open(ModelLayerInfoRelationId, AccessShareLock);
        if(!SearchSysCacheExists1(LAYERMODELNAME, CStringGetDatum(model_name))){
            ereport(ERROR,
                    errmsg("model \"%s\" does not exist in model_layer_info", model_name));
        }

        CatCList* parameter_list = SearchSysCacheList1(LAYERMODELNAME,CStringGetDatum(model_name));
        parm_vector_size = parameter_list->n_members;
        // layer_tensor_parm is ordered
        auto layer_tensor_parm = manager->module_handle_[model_path].first.named_parameters();

        // verify model layer num
        if(parm_vector_size != layer_tensor_parm.size()){
            ereport(ERROR,
                    errmsg("model \"%s\" layer num not equal to base model", model_name));
        }

        //int32 index=0;
        bool is_null;
        for(size_t index=0; index<parm_vector_size; ++index){
            HeapTuple proctup = &parameter_list->members[index]->tuple;
            char* layer_name = DatumGetCString(SysCacheGetAttr(LAYERMODELNAME, proctup, Anum_model_layer_info_layername, &is_null));


            for(const auto& parm : layer_tensor_parm){
                // model_layer_info exist in model
                //ereport(INFO,
                    //errmsg("parm.name:%s", parm.name.c_str()));
                if(parm.name == layer_name){
                    torch::Tensor tensor = parm.value.detach();
                    //ereport(INFO,
                        //errmsg("layer_name:%s", layer_name));
                    Vector* layer_parm = DatumGetVector(SysCacheGetAttr(LAYERMODELNAME, proctup, Anum_model_layer_info_parameter, &is_null));
                    tensor.copy_(vector_to_tensor(*layer_parm));
                    break;
                }
            }
        }
        ReleaseCatCacheList(parameter_list);
        table_close(pg_model_layer_info_rel, AccessShareLock);
        
    }
    return true;
    
}

char* replace_model_path(char* origin_path) {
    char                tmp_path[MAXPGPATH];
    char                model_path_root[MAXPGPATH];
    char                ret_path[MAXPGPATH];
    char                *ptr;

    strcpy(ret_path, origin_path);
    snprintf(model_path_root, MAXPGPATH, "%s/../model", pkglib_path);
    while ((ptr = strstr(ret_path, "{model_path}")) != NULL) {
        strcpy(tmp_path, ret_path);
        strlcpy(ret_path, tmp_path, ptr - ret_path + 1);
        strcat(ret_path, model_path_root);
        strcat(ret_path, tmp_path + (ptr - ret_path) + strlen("{model_path}"));
    }

    return pstrdup(ret_path);
}

bool 
model_manager_get_model_path(ModelManager *manager, const char *model_name, char **model_path, char **base_model)
{
    Relation            pg_model_info_rel; 
    Relation            pg_base_model_info_rel;
    HeapTuple           model_info_tuple;
    HeapTuple           base_model_info_tuple;
    Datum               path_datum;
    Datum               base_model_datum;
    bool                path_is_null;
    bool                base_model_is_null;
    pg_model_info_rel = table_open(ModelInfoRelationId, AccessShareLock);
    model_info_tuple = SearchSysCache1(MODELNAME, CStringGetDatum(model_name));

    Assert(model_info_tuple != NULL);

    if(!HeapTupleIsValid(model_info_tuple)){
        ereport(ERROR,
                (errcode(ERRCODE_DUPLICATE_MODEL),
                 errmsg("model \"%s\" not exists", model_name)));
        return false;
    }

    base_model_datum = heap_getattr(model_info_tuple, Anum_model_info_basemodel, 
                              RelationGetDescr(pg_model_info_rel), &base_model_is_null);    
    // have base model 
    if(!base_model_is_null){
        *base_model = DatumGetCString(base_model_datum);
        // model path read from base_model_info
        pg_base_model_info_rel = table_open(BaseModelInfoRelationId, AccessShareLock);
        base_model_info_tuple = SearchSysCache1(BASEMODEL, CStringGetDatum(*base_model));
        if(!HeapTupleIsValid(base_model_info_tuple)){
            ereport(ERROR,
                    (errcode(ERRCODE_UNDEFINED_BASE_MODEL),
                    errmsg("base model \"%s\" not exists", *base_model)));
            return false;
        }
        path_datum = heap_getattr(base_model_info_tuple, Anum_base_model_info_modelpath, 
                              RelationGetDescr(pg_base_model_info_rel), &path_is_null);
        if(!path_is_null) *model_path = replace_model_path(TextDatumGetCString(path_datum));  
        ReleaseSysCache(base_model_info_tuple);
        table_close(pg_base_model_info_rel, AccessShareLock);
    // new model 
    }else{
        path_datum = heap_getattr(model_info_tuple, Anum_model_info_modelpath, 
                              RelationGetDescr(pg_model_info_rel), &path_is_null);
        if(!path_is_null) *model_path = replace_model_path(TextDatumGetCString(path_datum));  
    }

    ReleaseSysCache(model_info_tuple);
    table_close(pg_model_info_rel, AccessShareLock);

    return true;
}

bool 
model_manager_get_model_md5(ModelManager *manager, const char *model_path, char **md5)
{
    Relation            pg_model_info_rel; 
    HeapTuple           tuple;
    Datum               md5_datum;
    bool                md5_is_null;

    if(!SearchSysCacheExists1(Anum_model_info_modelname, CStringGetDatum(model_path))){
        ereport(ERROR,
                (errcode(ERRCODE_DUPLICATE_MODEL),
                 errmsg("model \"%s\" not exists", model_path)));
        return false;
    }

    pg_model_info_rel = table_open(ModelInfoRelationId, AccessShareLock);

    tuple = SearchSysCache1(Anum_model_info_modelpath, CStringGetDatum(model_path));

    Assert(tuple != NULL);

    md5_datum = heap_getattr(tuple, Anum_model_info_md5, 
                              RelationGetDescr(pg_model_info_rel), &md5_is_null);

    if(!md5_is_null) *md5 = TextDatumGetCString(md5_datum);

    table_close(pg_model_info_rel, AccessShareLock);

    return true;
}

bool 
model_manager_set_cuda(ModelManager *manager, const char *model_path)
{
    if(manager->module_handle_.find(model_path) != manager->module_handle_.end()){
        if(manager->module_handle_[model_path].second == at::kCUDA){
            return true;
        } else {
            if(torch::cuda::is_available()){
                manager->module_handle_[model_path].second = at::kCUDA;
                manager->module_handle_[model_path].first.to(at::kCUDA);
                manager->module_handle_[model_path].first.eval();
                ereport(INFO, (errmsg("%s use gpu!", model_path)));
                return true;
            }
            return false;
        }
    } else {
        return false;
    }
}

bool 
model_manager_get_device_type(ModelManager *manager, const char *model_path, torch::DeviceType& device_type)
{
    if(manager->module_handle_.find(model_path) == manager->module_handle_.end()){
        return false;
    }
    device_type = manager->module_handle_[model_path].second;
    return true;
}

bool 
model_manager_pre_process(ModelManager *manager, const char *model_path, std::vector<torch::jit::IValue>& input_tensor, Args *args)
{
    if(manager->module_preprocess_functions_.find(model_path) != manager->module_preprocess_functions_.end()){
model_process_registered:
        if(manager->module_preprocess_functions_[model_path](input_tensor, args)){
            if(manager->module_handle_.find(model_path) != manager->module_handle_.end()){
                for(auto& tensor : input_tensor){
                    tensor = tensor.toTensor().to(manager->module_handle_[model_path].second);
                }
                return true;
            }else{
                ereport(ERROR, (errmsg("model:%s handle not exist!", model_path)));
            }
        }else{
            return false;
        }
        
    } else {
        register_default_model();
        goto model_process_registered;
    }
    return false;
}

bool 
model_manager_output_process_float(ModelManager *manager, const char *model_path, torch::jit::IValue& output_tensor, Args *args, float8& result)
{
    if(manager->module_outputprocess_functions_float_.find(model_path) != manager->module_outputprocess_functions_float_.end()){
        return manager->module_outputprocess_functions_float_[model_path](output_tensor, args, result);
    }

    return false;
}

bool 
model_manager_output_process_text(ModelManager *manager, const char *model_path, torch::jit::IValue& output_tensor, Args *args, std::string& result)
{
    if(manager->module_outputprocess_functions_text_.find(model_path) != manager->module_outputprocess_functions_text_.end()){
        //ereport(INFO, (errmsg("user output_text function")));
        return manager->module_outputprocess_functions_text_[model_path](output_tensor, args, result);
    }
    // auto max_result = output_tensor.max(1,true);
    // auto max_index = std::get<1>(max_result).item<float>();
    // char buffer[50];
    // sprintf(buffer, "%f", max_index);
    // result = cstring_to_text("12123");
    return false;
}

void 
model_manager_register_pre_process(ModelManager *manager, const char *model_name, PreProcessCallback func)
{
    char* model_path = nullptr;
    char* base_model = nullptr;
    if(model_manager_get_model_path(manager, model_name, &model_path, &base_model)){
        manager->module_preprocess_functions_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name)));
    }
}

void 
model_manager_register_output_process_float(ModelManager *manager, const char *model_name, OutputProcessFloatCallback func)
{
    char* model_path = nullptr;
    char* base_model = nullptr;
    if(model_manager_get_model_path(manager, model_name, &model_path, &base_model)){
        manager->module_outputprocess_functions_float_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name)));
    }
}

void 
model_manager_register_output_process_text(ModelManager *manager, const char *model_name, OutputProcessTextCallback func)
{
    char* model_path = nullptr;
    char* base_model = nullptr;
    if(model_manager_get_model_path(manager, model_name, &model_path, &base_model)){
        manager->module_outputprocess_functions_text_[model_path] = func;
        return;
    }else{
        ereport(ERROR, (errmsg("model:%s not exist!", model_name)));
    }
}

bool 
model_manager_predict(ModelManager *manager, const char *model_path, torch::jit::IValue& input, torch::jit::IValue& output)
{
    if(manager->module_handle_.find(model_path) == manager->module_handle_.end()){
        return false;
    }
    try {
        output = manager->module_handle_[model_path].first.forward({input}).toTensor();
        return true;
    }
    catch (const std::exception& e) {
        ereport(ERROR, (errmsg("predict error, error message:%s", e.what())));
        return false;
    }
}

bool 
model_manager_predict_multi_input(ModelManager *manager, const char *model_path, std::vector<torch::jit::IValue>& input, torch::jit::IValue& output)
{
    if(manager->module_handle_.find(model_path) == manager->module_handle_.end()){
        return false;
    }
    try {
        output = manager->module_handle_[model_path].first.forward(input);
        return true;
    }
    catch (const std::exception& e) {
        ereport(ERROR, (errmsg("muti predict error, error message:%s", e.what())));
        return false;
    }
}

#ifdef __cplusplus
}
#endif
