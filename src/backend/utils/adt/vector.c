#include "postgres.h"

#include "catalog/pg_type_d.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/palloc.h"
#include "utils/vector.h"
#include <stdbool.h>

#define INIT_VECTOR

MVec *
new_mvec(int dim, int shape_size)
{
    MVec *new_col = (MVec *)palloc(GET_MVEC_SIZE(dim));
    if (shape_size == 0)
    {
        shape_size = 1;
        SET_MVEC_DIM_SHAPESIZE(new_col, dim, shape_size);
        SET_MVEC_SHAPE_VAL(new_col, 0, 0);
    }
    else
    {
        SET_MVEC_DIM_SHAPESIZE(new_col, dim, shape_size);
    }

    return new_col;
}

MVec *
new_mvec_ref(RowId row_id)
{
    MVec *new_col = (MVec *)palloc(MVEC_POINTER_SIZE);
    SET_MVEC_REF_ROWID(new_col, row_id);
    return new_col;
}

void free_vector(MVec *vector)
{
    if (vector == NULL)
    {
        return;
    }
    pfree(vector);
    vector = NULL;
}

static inline bool
is_space(char ch)
{
    if (ch == ' ' ||
        ch == '\t' ||
        ch == '\n' ||
        ch == '\r' ||
        ch == '\v' ||
        ch == '\f')
        return true;
    return false;
}

static inline void
skip_space(char **p_str)
{
    char *p_ch = *p_str;
    char ch = *p_ch;
    while (ch == ' ' ||
           ch == '\t' ||
           ch == '\n' ||
           ch == '\r' ||
           ch == '\v' ||
           ch == '\f')
    {
        p_ch++;
        ch = *p_ch;
    }
    *p_str = p_ch;
}

static const char *
generate_space(int num)
{
    int i = 0;
    StringInfoData ret;
    initStringInfo(&ret);
    for (i = 0; i < num; i++)
        appendStringInfoChar(&ret, ' ');
    return ret.data;
}

static void
print_parse_error(const char *err_msg, const char *whole_str, const char *err_pos)
{
    const char *p_last_20 = (err_pos - 20);
    int last_words = (int)(p_last_20 - whole_str);
    if (last_words < 0)
    {
        p_last_20 = whole_str;
        last_words = err_pos - p_last_20;
    }
    const char *p_forward_20 = pnstrdup(err_pos, 20);
    int err_word_pos = (int)(err_pos - whole_str);

    bool need_prefix = p_last_20 != whole_str;
    bool need_suffix = strlen(p_forward_20) != strlen(err_pos);
    const char *hint_prefix = "error occur at pos (";
    int space_num = strlen(hint_prefix) + strlen("HINT:  ") + 5 + strlen("): \"") + last_words;
    if (need_prefix)
        space_num += 3;
    ereport(ERROR,
            (errmsg("invalid input: \"%s\": %s.", whole_str, err_msg),
             errhint("error occur at pos (%5d): \"%s%s%s%s\"\n%s^",
                     err_word_pos,
                     need_prefix ? "..." : "",
                     pnstrdup(p_last_20, last_words), p_forward_20,
                     need_suffix ? "..." : "",
                     generate_space(space_num))));
}

/*
    parse vector shape
*/
static inline void
parse_vector_shape_str(char *shape_str, unsigned int *shape_size, int *shape)
{
    char *str_copy = pstrdup(shape_str);
    char *pt = NULL;
    char *end = NULL;

    while (is_space(*shape_str))
    {
        shape_str++;
    }

    if (*shape_str != '{')
    {
        ereport(ERROR,
                errmsg("Vector shape must start with \"{\"."));
    }

    shape_str++;
    pt = strtok(shape_str, ",");
    end = pt;

    while (pt != NULL && *end != '}')
    {
        if (*shape_size == MAX_VECTOR_SHAPE_SIZE)
        {
            ereport(ERROR,
                    (errmsg("vector shape cannot have more than %d", MAX_VECTOR_SHAPE_SIZE)));
        }

        while (is_space(*pt))
        {
            pt++;
        }

        if (*pt == '\0')
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        shape[*shape_size] = strtof(pt, &end);

        (*shape_size)++;

        if (end == pt)
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        while (is_space(*end))
        {
            end++;
        }

        if (*end != '\0' && *end != '}')
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector shape: \"%s\"", str_copy)));
        }

        pt = strtok(NULL, ",");
    }

    if (end == NULL || *end != '}')
    {
        ereport(ERROR,
                (errmsg("malformed vector literal4: \"%s\"", str_copy)));
    }

    end++;

    while (is_space(*end))
    {
        end++;
    }

    for (pt = str_copy + 1; *pt != '\0'; pt++)
    {
        if (pt[-1] == ',' && *pt == ',')
        {
            ereport(ERROR,
                    (errmsg("malformed vector literal5: \"%s\"", str_copy)));
        }
    }

    if (*shape_size < 1)
    {
        ereport(ERROR,
                (errmsg("vector must have at least 1 dimension")));
    }

    pfree(str_copy);
}

static inline void
parse_vector_str(char *str, unsigned int *dim, float *x,
                 unsigned int *shape_size, int32 *shape)
{
    char *str_copy = pstrdup(str);
    char *index = str_copy;
    char *pt = NULL;
    char *end = NULL;

    while (is_space(*str))
    {
        str++;
    }

    if (*str != '[')
    {
        ereport(ERROR,
                errmsg("Vector contents must start with \"[\"."));
    }

    str++;
    pt = strtok(str, ",");
    end = pt;

    while (pt != NULL && *end != ']')
    {
        if (*dim == MAX_VECTOR_DIM)
        {
            ereport(ERROR,
                    (errmsg("vector cannot have more than %d dimensions", MAX_VECTOR_DIM)));
        }

        while (is_space(*pt))
        {
            pt++;
        }

        if (*pt == '\0')
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        x[*dim] = strtof(pt, &end);

        (*dim)++;

        if (end == pt)
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        while (is_space(*end))
        {
            end++;
        }

        if (*end != '\0' && *end != ']')
        {
            ereport(ERROR,
                    (errmsg("invalid input syntax for type vector: \"%s\"", str_copy)));
        }

        pt = strtok(NULL, ",");
    }

    if (end == NULL || *end != ']')
    {
        ereport(ERROR,
                (errmsg("malformed vector literal3: \"%s\"", str_copy)));
    }

    end++;

    while (is_space(*end))
    {
        end++;
    }

    switch (*end)
    {
    case '{':
        while (*str_copy != '{')
        {
            str_copy++;
        }
        parse_vector_shape_str(str_copy, shape_size, shape);
        str_copy = index;
        break;
    case '\0':
        *shape_size = 1;
        shape[0] = (int32)(*dim);
        break;
    default:
        ereport(ERROR,
                (errmsg("malformed vector literal1: \"%s\"", str_copy)));
    }

    for (pt = str_copy + 1; *pt != '\0'; pt++)
    {
        if (pt[-1] == ',' && *pt == ',')
        {
            ereport(ERROR,
                    (errmsg("malformed vector literal2: \"%s\"", str_copy)));
        }
    }

    if (*dim < 1)
    {
        ereport(ERROR,
                (errmsg("vector must have at least 1 dimension")));
    }

    pfree(str_copy);
}

static inline bool
shape_equal(MVec *left, MVec *right)
{
    if (left == NULL || right == NULL)
    {
        return false;
    }

    if (GET_MVEC_SHAPE_SIZE(left) != GET_MVEC_SHAPE_SIZE(right))
    {
        return false;
    }

    for (int i = 0; i < GET_MVEC_SHAPE_SIZE(left); ++i)
    {
        if (GET_MVEC_SHAPE_VAL(left, i) != GET_MVEC_SHAPE_VAL(right, i))
        {
            return false;
        }
    }
    return true;
}

Datum mvec_input(PG_FUNCTION_ARGS)
{
    char *str = NULL;
    float *x = (float *)palloc(sizeof(float) * MAX_VECTOR_DIM);
    int32 shape[MAX_VECTOR_SHAPE_SIZE];
    MVec *vector = NULL;
    unsigned int dim = 0;
    unsigned int shape_size = 0;
    unsigned int shape_dim = 1;

    str = PG_GETARG_CSTRING(0);

    parse_vector_str(str, &dim, x, &shape_size, shape);

    // Verify that the multiplication of shape values equals dim
    for (int i = 0; i < shape_size; i++)
    {
        shape_dim *= shape[i];
    }

    if (dim != shape_dim)
    {
        ereport(ERROR,
                (errmsg("the multiplication of shape values not equals, dim:%d, shape_dim:%d", dim, shape_dim)));
    }
    vector = new_mvec(dim, shape_size);

    for (int i = 0; i < dim; ++i)
    {
        SET_MVEC_VAL(vector, i, x[i]);
    }

    for (int i = 0; i < shape_size; ++i)
    {
        SET_MVEC_SHAPE_VAL(vector, i, shape[i]);
    }

    pfree(x);
    PG_RETURN_POINTER(vector);
}

Datum mvec_output(PG_FUNCTION_ARGS)
{
    MVec *mvec = NULL;
    StringInfoData ret;
    int32 dim = 0;
    int32 shape_size = 0;

    mvec = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(mvec);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec);

    if (dim > MAX_VECTOR_DIM)
    {
        ereport(ERROR,
                (errmsg("dim is larger than %d dim!", MAX_VECTOR_DIM)));
    }

    if (shape_size > MAX_VECTOR_SHAPE_SIZE)
    {
        ereport(ERROR,
                (errmsg("shape size is larger than 10!")));
    }

    initStringInfo(&ret);
    appendStringInfoChar(&ret, '[');

    if (dim > 10)
    {
        for (int i = 0; i < 3; ++i)
        {
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            appendStringInfoChar(&ret, ',');
        }
        appendStringInfoString(&ret, "....");
        appendStringInfoChar(&ret, ',');
        for (int i = dim - 3; i < dim; ++i)
        {
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            if (i != dim - 1)
            {
                appendStringInfoChar(&ret, ',');
            }
        }
    }
    else
    {
        for (int i = 0; i < dim; ++i)
        {
            appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(float4out, Float4GetDatum(GET_MVEC_VAL(mvec, i)))));
            if (i != dim - 1)
            {
                appendStringInfoChar(&ret, ',');
            }
        }
    }

    appendStringInfoChar(&ret, ']');

    appendStringInfoChar(&ret, '{');
    for (int i = 0; i < shape_size; ++i)
    {
        appendStringInfoString(&ret, DatumGetCString(DirectFunctionCall1(int4out, Int32GetDatum(GET_MVEC_SHAPE_VAL(mvec, i)))));
        if (i != shape_size - 1)
        {
            appendStringInfoChar(&ret, ',');
        }
    }
    appendStringInfoChar(&ret, '}');

    PG_RETURN_CSTRING(ret.data);
}

Datum mvec_receive(PG_FUNCTION_ARGS)
{
    StringInfo str;
    MVec *ret = NULL;
    int32_t dim = 0;
    int32_t shape_size = 0;

    str = (StringInfo)PG_GETARG_POINTER(0);
    dim = pq_getmsgint(str, sizeof(int32_t));

    ret = new_mvec(dim, 1);
    for (int i = 0; i < dim; ++i)
    {
        SET_MVEC_VAL(ret, i, pq_getmsgfloat4(str));
    }

    PG_RETURN_POINTER(ret);
}

Datum mvec_send(PG_FUNCTION_ARGS)
{
    MVec *mvec;
    StringInfoData str;

    mvec = PG_GETARG_MVEC_P(0);

    pq_begintypsend(&str);
    pq_sendint(&str, GET_MVEC_DIM(mvec), sizeof(int32_t));
    for (int i = 0; i < GET_MVEC_DIM(mvec); ++i)
    {
        pq_sendfloat4(&str, GET_MVEC_VAL(mvec, i));
    }

    PG_RETURN_BYTEA_P(pq_endtypsend(&str));
}

Datum array_to_mvec(PG_FUNCTION_ARGS)
{
    ArrayType *array = NULL;
    Oid array_type;
    MVec *mvec = NULL;
    int array_length;
    Datum *elems = NULL;
    bool *nulls = NULL;

    array = PG_GETARG_ARRAYTYPE_P(0);

    array_type = ARR_ELEMTYPE(array);

    switch (array_type)
    {
    case FLOAT4OID:
    {
        deconstruct_array(array, FLOAT4OID, sizeof(float4), true, 'i', &elems, &nulls, &array_length);
        break;
    }
    case FLOAT8OID:
    {
        deconstruct_array(array, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd', &elems, &nulls, &array_length);
        for (int i = 0; i < array_length; ++i)
        {
            elems[i] = Float4GetDatum(DatumGetFloat8(elems[i]));
        }
        break;
    }
    case INT4OID:
    {
        deconstruct_array(array, INT4OID, sizeof(int), true, 'i', &elems, &nulls, &array_length);
        for (int i = 0; i < array_length; ++i)
        {
            elems[i] = Float4GetDatum(DatumGetInt32(elems[i]));
        }
        break;
    }
    default:
    {
        ereport(ERROR,
                (errmsg("unsupport %d type to mvec!"), array_type));
    }
    }

    if (array_length > MAX_VECTOR_DIM)
    {
        ereport(ERROR,
                (errmsg("mvec cannot have more than %d dimensions", MAX_VECTOR_DIM)));
    }

    mvec = new_mvec(array_length, 1);

    for (int i = 0; i < array_length; ++i)
    {
        SET_MVEC_VAL(mvec, i, DatumGetFloat4(elems[i]));
    }

    SET_MVEC_SHAPE_VAL(mvec, 0, array_length);

    PG_RETURN_POINTER(mvec);
}

Datum mvec_to_float_array(PG_FUNCTION_ARGS)
{
    MVec *mvec = NULL;
    ArrayType *ret = NULL;
    Datum *elems = NULL;
    int32_t dim = 0;

    mvec = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(mvec);

    elems = (Datum *)palloc(sizeof(Datum *) * dim);

    for (int i = 0; i < dim; ++i)
    {
        elems[i] = Float4GetDatum(GET_MVEC_VAL(mvec, i));
    }

    ret = construct_array(elems, dim, FLOAT4OID, sizeof(float4), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(ret);
}

Datum text_to_mvec(PG_FUNCTION_ARGS)
{
    char *str = NULL;
    MVec *vector = NULL;
    unsigned int dim = 0;

    int32 shape[MAX_VECTOR_SHAPE_SIZE];
    float *x = (float *)palloc(sizeof(float) * MAX_VECTOR_DIM);
    unsigned int shape_size = 0;
    unsigned int shape_dim = 1;

    str = TextDatumGetCString(PG_GETARG_DATUM(0));

    parse_vector_str(str, &dim, x, &shape_size, shape);

    for (int i = 0; i < shape_size; i++)
    {
        shape_dim *= shape[i];
    }

    if (dim != shape_dim)
    {
        ereport(ERROR,
                (errmsg("the multiplication of shape values not equals, dim:%d, shape_dim:%d", dim, shape_dim)));
    }

    vector = new_mvec(dim, shape_size);

    for (int i = 0; i < dim; ++i)
    {
        SET_MVEC_VAL(vector, i, x[i]);
    }

    for (int i = 0; i < shape_size; ++i)
    {
        SET_MVEC_SHAPE_VAL(vector, i, shape[i]);
    }

    pfree(x);
    PG_RETURN_POINTER(vector);
}

Datum get_mvec_data(PG_FUNCTION_ARGS)
{
    MVec *vector = NULL;
    // float           x[MAX_VECTOR_DIM];
    ArrayType *result = NULL;
    Datum *elems = NULL;
    unsigned int dim = 0;

    vector = PG_GETARG_MVEC_P(0);

    dim = GET_MVEC_DIM(vector);

    elems = (Datum *)palloc(sizeof(Datum *) * dim);

    for (int i = 0; i < dim; ++i)
    {
        elems[i] = Float4GetDatum(GET_MVEC_VAL(vector, i));
    }

    result = construct_array(elems, dim, FLOAT4OID, sizeof(float4), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum get_mvec_shape(PG_FUNCTION_ARGS)
{
    MVec *vector = NULL;
    // float           x[MAX_VECTOR_DIM];
    ArrayType *result = NULL;
    Datum *elems = NULL;
    unsigned int shape_size = 0;

    vector = PG_GETARG_MVEC_P(0);

    shape_size = GET_MVEC_SHAPE_SIZE(vector);

    elems = (Datum *)palloc(sizeof(Datum *) * shape_size);

    for (int i = 0; i < shape_size; ++i)
    {
        elems[i] = Int32GetDatum(GET_MVEC_SHAPE_VAL(vector, i));
    }

    result = construct_array(elems, shape_size, INT4OID, sizeof(int32), true, 'i');

    pfree(elems);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum mvec_add(PG_FUNCTION_ARGS)
{
    MVec *mvec_left = NULL;
    MVec *mvec_right = NULL;
    MVec *ret = NULL;
    int32_t dim = 0;
    int32_t shape_size = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);

    if (GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right))
    {
        ereport(ERROR,
                (errmsg("the two mvecs have different dimensions!,(),()")));
    }

    if (!shape_equal(mvec_left, mvec_right))
    {
        ereport(ERROR,
                (errmsg("the two mvecs have different shape!,(),()")));
    }

    dim = GET_MVEC_DIM(mvec_left);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec_left);
    ret = new_mvec(dim, shape_size);

    for (int i = 0; i < dim; ++i)
    {
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        float res = left + right;
        if (unlikely(isinf(res)) && !isinf(left) && !isinf(right))
        {
            ereport(ERROR,
                    (errmsg("overflow for %f + %f", left, right)));
        }
        SET_MVEC_VAL(ret, i, (left + right));
    }

    for (int i = 0; i < shape_size; ++i)
    {
        int32_t value = GET_MVEC_SHAPE_VAL(mvec_left, i);
        SET_MVEC_SHAPE_VAL(ret, i, value);
    }

    PG_RETURN_POINTER(ret);
}

Datum mvec_sub(PG_FUNCTION_ARGS)
{
    MVec *mvec_left = NULL;
    MVec *mvec_right = NULL;
    MVec *ret = NULL;
    int32_t dim = 0;
    int32_t shape_size = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);

    if (GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right))
    {
        ereport(ERROR,
                (errmsg("the two mvecs have different dimensions!,(),()")));
    }

    if (!shape_equal(mvec_left, mvec_right))
    {
        ereport(ERROR,
                (errmsg("the two mvecs have different shape!,(),()")));
    }

    dim = GET_MVEC_DIM(mvec_left);
    shape_size = GET_MVEC_SHAPE_SIZE(mvec_left);
    ret = new_mvec(dim, shape_size);

    for (int i = 0; i < dim; ++i)
    {
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        float res = left - right;
        if (unlikely(isinf(res)) && !isinf(left) && !isinf(right))
        {
            ereport(ERROR,
                    (errmsg("overflow for %f - %f", left, right)));
        }
        SET_MVEC_VAL(ret, i, res);
    }

    for (int i = 0; i < shape_size; ++i)
    {
        int32_t value = GET_MVEC_SHAPE_VAL(mvec_left, i);
        SET_MVEC_SHAPE_VAL(ret, i, value);
    }

    PG_RETURN_POINTER(ret);
}

Datum mvec_equal(PG_FUNCTION_ARGS)
{
    MVec *mvec_left = NULL;
    MVec *mvec_right = NULL;
    MVec *ret = NULL;
    int32_t dim = 0;

    mvec_left = PG_GETARG_MVEC_P(0);
    mvec_right = PG_GETARG_MVEC_P(1);

    if (GET_MVEC_DIM(mvec_left) != GET_MVEC_DIM(mvec_right))
    {
        PG_RETURN_BOOL(false);
    }

    dim = GET_MVEC_DIM(mvec_left);

    for (int i = 0; i < dim; ++i)
    {
        float left = GET_MVEC_VAL(mvec_left, i);
        float right = GET_MVEC_VAL(mvec_right, i);
        if (isnan(left) ? !isnan(right)
                        : (isnan(right) || left - right > 1e-6 || right - left > 1e-6))
            PG_RETURN_BOOL(false);
    }

    PG_RETURN_BOOL(true);
}

// void mvec_to_str(MVec *mvec, std::string &str)
// {
//     if (mvec == NULL)
//     {
//         str = "";
//         return;
//     }
//     str += '[';
//     for (int i = 0; i < GET_MVEC_DIM(mvec); i++)
//     {
//         str += std::to_string(GET_MVEC_VAL(mvec, i));
//         if (i != (GET_MVEC_DIM(mvec) - 1))
//         {
//             str += ',';
//         }
//     }
//     str += ']';
//     str += '{';
//     for (int i = 0; i < GET_MVEC_SHAPE_SIZE(mvec); i++)
//     {
//         str += std::to_string(GET_MVEC_SHAPE_VAL(mvec, i));
//         if (i != (GET_MVEC_SHAPE_SIZE(mvec) - 1))
//         {
//             str += ',';
//         }
//     }
//     str += '}';
// }