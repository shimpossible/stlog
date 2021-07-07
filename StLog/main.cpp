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

AllocBackEnd be;
std::atomic<int> i = 0;
std::atomic<float> k2 = 0;

LogProvider* pro = 0;
void PushThread(void *arg)
{
	int loops = (int)arg;
	Logger& logger = pro->get("test");
	Logger& log2 = pro->get("log2");


	for(int k=0;k<loops;k++)
	{
		AttributeValue val(GetCurrentThreadId());
		LogScope scope = logger.begin_scope({
			{"file.name", __FILE__},
			{"file.line", (int)__LINE__},
			{"thread.id", val },
			});

		int _i = i;
		logger.log(Severity::Info, "Hello World",
			{
				{"key1", (int)i},
				{"key2", k2.load()},
				{"key3", "asdf" + std::to_string(k2) + " " + std::to_string(i)},
			}
		);
		i+=1;

		k2.store(k2 + 0.001);
		{
			LogScope scop2 = log2.begin_scope({
				{"inner", "inner"},
				});
			log2.log(Severity::Info, "Nexted", {});
		}
		scope.end();
		Sleep(100);
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

	pro = new LogProvider();
	PrintfLogExporter* ex = new PrintfLogExporter(be);
	SimpleLogProcessor* p = new SimpleLogProcessor(*pro, be);

	pro->with_exporter(ex)
		.with_processor(p);


	//PushThread((void*)1000);
	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);

	while (true)
	{
		Sleep(1000);
	}
}
