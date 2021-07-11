//#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC_NEW
#include <crtdbg.h>
#include <assert.h>
//#endif

#include <process.h>
#include "LogRecord.h"
#include <process.h>
#include <Windows.h>
#include <iostream>
#include <string>

thread_local LogScope* Logger::m_scope = 0;

AllocBackEnd be({
	{16,  2048, (char*)malloc(16 * 2048) },
	{64,  4096, (char*)malloc(64 * 4096) },
	{128, 512,  (char*)malloc(128 * 512) },
	{256, 512,  (char*)malloc(256 * 512) },
});
std::atomic<int> i = 0;
std::atomic<float> k2 = 0;

LogProvider* pro = 0;

std::atomic<int64_t> avg;
void PushThread(void *arg)
{
	int loops = (int)arg;
	Logger& logger = pro->get("test");
	Logger& log2 = pro->get("log2");

	volatile char buff[1024];

	for(int k=0;k<loops;k++)
	{
		AttributeValue val(GetCurrentThreadId());
		LogScope scope = logger.begin_scope({
			{"file.name", __FILE__},
			{"file.line", (int)__LINE__},
			{"thread.id", val },
			});

		int _i = i;
		while (!i.compare_exchange_weak(_i, _i + 1))
			_i = i;

		float _k2 = k2;
		while (!k2.compare_exchange_weak(_k2, _k2 + 0.001f))
			_k2 = k2;

		auto t1 = std::chrono::steady_clock::now();

//#define SPRINT	

		std::string s = "asdf" + std::to_string(k2) + " " + std::to_string(i);

		for (int i = 0; i < 100; i++)
		{
#ifdef SPRINT
			uint64_t ts = std::chrono::system_clock::now().time_since_epoch().count();
			sprintf((char*)buff, "%llu Hello World %s: %d, %s, %f %s: %s %s: %s, %s: %d, %s: %d\n",
				ts,
				"key1", _i,
				"key2", k2.load(),
				"key3", s.c_str(),
				"file.name", __FILE__,
				"file.line", __LINE__,
				"thread.id", val
			);
#else		

			logger.log(Severity::Info, "Hello World",
				{
					{"key1", (int)_i},
					{"key2", k2.load()},
					{"key3", s},
				}
			);
#endif
		}
		auto t2 = std::chrono::steady_clock::now();

		auto diff = t2 - t1;
		int64_t a = avg;
		while (!avg.compare_exchange_weak(a, a + (diff.count() - a) * 0.01))
		{
			a = avg;
		}

		{
			LogScope scop2 = log2.begin_scope({
				{"inner", "inner value"},
				});
			log2.log(Severity::Info, "Nested", {});
		}
		scope.end();
		//Sleep(1);
	}
}

void *p1, *p2, *p3;
int main()
{
	AttributeValue v(0);

	/*
	_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG
	| _CRTDBG_LEAK_CHECK_DF
	| _CRTDBG_ALLOC_MEM_DF
	| _CRTDBG_CHECK_ALWAYS_DF);
	*/

	//_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
	//_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
	_CrtSetDbgFlag(
		  _CRTDBG_ALLOC_MEM_DF 
		| _CRTDBG_LEAK_CHECK_DF
		| _CRTDBG_CHECK_ALWAYS_DF
		| _CRTDBG_CHECK_CRT_DF
	    | _CRTDBG_REPORT_FLAG);


	AttributeValue av(0);
	//std::pair<int, AttributeValue> pair(1, av);

	printf("starting...");

	/*
	* // Check for leaks
	_CrtMemState state, state2;
	_CrtMemCheckpoint(&state);
	PushThread(1000);
	_CrtMemCheckpoint(&state2);
	_CrtMemDumpAllObjectsSince(&state);
	*/

	FILE* f;
	fopen_s(&f, "log.txt", "w");
	pro = new LogProvider();
	PrintfLogExporter  ex(stdout);
	NoAllocLogProcessor p(*pro, be);

	pro->with_exporter(&ex)
		.with_processor(&p)
		;


	//PushThread((void*)1000);
	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);

	for (int i = 0; i < 20; i++)
	{
		Sleep(1000);
		printf("%llu\n", avg.load());
	}
	Sleep(10000);

	pro->with_processor(nullptr);
}
