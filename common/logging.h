// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: logging

#pragma once

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define get_thread_id() GetCurrentThreadId()
#else
#include <sys/syscall.h>
#define get_thread_id() syscall(SYS_gettid)
#endif

#include <time.h>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

inline char *curr_time() {
    time_t raw_time = time(nullptr);
    struct tm *time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S", time_info)] = '\0';

    return now_time;
}

inline int get_pid() {
    static int pid = getpid();

    return pid;
}

inline long int get_tid() {
    thread_local long int tid = get_thread_id();

    return tid;
}

#define HGEMM_LOG_TAG "HGEMM"
#define HGEMM_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                                                         \
    do {                                                                                                          \
        fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", HGEMM_LOG_TAG, curr_time(), get_pid(), get_tid(), \
                HGEMM_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__);                                 \
    } while (0)
