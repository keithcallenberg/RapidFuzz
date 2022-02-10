#pragma once
#include "cpp_process.hpp"
#include "numpy/ndarraytypes.h"
#include "taskflow/taskflow.hpp"
#include "rapidfuzz_capi.h"
#include <exception>
#include <atomic>

int64_t any_round(double score)
{
    return std::llround(score);
}

int64_t any_round(int64_t score)
{
    return score;
}

template <typename T>
void set_score(PyArrayObject* array, npy_intp row, T score)
{
    void* data = PyArray_GETPTR1(array, row);
    *((int64_t*)data) = any_round(score);
}

template <typename Func>
void run_parallel(int workers, int64_t rows, Func&& func)
{
    /* for these cases spawning threads causes too much overhead to be worth it */
    if (workers == 0 || workers == 1)
    {
        func(0, rows);
        return;
    }

    if (workers < 0)
    {
        workers = std::thread::hardware_concurrency();
    }

    std::exception_ptr exception = nullptr;
    std::atomic<int> exceptions_occurred{0};
    tf::Executor executor(workers);
    tf::Taskflow taskflow;
    std::int64_t step_size = 1;

    taskflow.for_each_index((std::int64_t)0, rows, step_size, [&] (std::int64_t row) {
        /* skip work after an exception occurred */
        if (exceptions_occurred.load() > 0) {
            return;
        }
        try
        {
            std::int64_t row_end = std::min(row + step_size, rows);
            func(row, row_end);
        }
        catch(...)
        {
            /* only store first exception */
            if (exceptions_occurred.fetch_add(1) == 0) {
                exception = std::current_exception();
            }
        }
    });

    executor.run(taskflow).get();

    if (exception) {
        std::rethrow_exception(exception);
    }
}

template <typename T>
static PyObject* eofm_two_lists_impl(
    const RF_Kwargs* kwargs, RF_Scorer* scorer,
    const std::vector<RF_StringWrapper>& queries, const std::vector<RF_StringWrapper>& choices, int dtype, int workers, T score_cutoff)
{
    std::int64_t rows = queries.size();
    std::int64_t cols = choices.size();
    npy_intp dims[] = {(npy_intp)rows};
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(1, dims, int64_t);

    if (array == NULL)
    {
        return array;
    }

    std::exception_ptr exception = nullptr;

Py_BEGIN_ALLOW_THREADS
    try
    {
        run_parallel(workers, rows, [&] (std::int64_t row, std::int64_t row_end) {
            for (; row < row_end; ++row)
            {
                RF_ScorerFunc scorer_func;
                PyErr2RuntimeExn(scorer->scorer_func_init(&scorer_func, kwargs, 1, &queries[row].string));
                RF_ScorerWrapper ScorerFunc(scorer_func);
                float max = 0;
                std::int64_t argmax = NULL;
                for (int64_t col = 0; col < cols; ++col)
                {
                    T score;
                    ScorerFunc.call(&choices[col].string, score_cutoff, &score);
                    if (score >= max) {
                        max = score;
                        argmax = col;
                    }
                }
                set_score(array, row, argmax);
            }
        });
    }
    catch(...)
    {
        exception = std::current_exception();
    }
Py_END_ALLOW_THREADS

    if (exception) {
        std::rethrow_exception(exception);
    }

    return (PyObject*)array;
}
