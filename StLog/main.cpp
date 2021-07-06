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

struct AttrPrinter
{
	void operator()(bool b) 
	{
		printf("%d", b);
	}

	void operator()(float b)
	{
		printf("%f", b);
	}
	void operator()(double b)
	{
		printf("%lf", b);
	}
	void operator()(int8_t b)
	{
		printf("%d", b);
	}
	void operator()(int16_t b)
	{
		printf("%d", b);
	}
	void operator()(int32_t b)
	{
		printf("%d", b);
	}
	void operator()(int64_t b) 
	{
		printf("%lld", b);
	}

	void operator()(uint8_t b) 
	{
		printf("%u", b);
	}
	void operator()(uint16_t b) 
	{
		printf("%u", b);
	}
	void operator()(uint32_t b) 
	{
		printf("%u", b);
	}
	void operator()(uint64_t b) 
	{
		printf("%llu", b);
	}

	void operator()(const Span<const char>& b)
	{
		printf("%.*s", (int)b.len, b.data);
	}
};

// single buffer shared by ALL loggers
RingBuffer* buff_ptr;

void ReadThread(void* )
{
	AllocBackEnd be;
	SimpleLogExporter e(be, *buff_ptr);
	std::cout << "START\n";
	printf("READ THREAD STARTED\n");
	while (true)
	{
		SimpleLogRecord slr;

		char buff[4096];
		bool b1 = e.read_record(slr, buff, sizeof(buff));
		if (b1)
		{
			printf("%llu ", slr.ts);
			printf("%04x ", slr.severity);
			printf("%.*s - ", (int)slr.name.len, slr.name.data);
			printf("%.*s - ", (int)slr.message.len, slr.message.data);
			for (auto it = slr.m_attrs.begin();
				      it != slr.m_attrs.end();
				      it++)
			{
				printf("\n\t%s : ", it->first.c_str());

				if (it->first == "file.name")
				{
					size_t n = strlen(__FILE__);
					if (strncmp(it->second.data.s.data, __FILE__, n) != 0 ||
						n != it->second.data.s.len)
					{
						printf("WRONG STRING\n");
					}
				}
				AttrPrinter ptr;
				visit(ptr, it->second);
			};
			printf("\n");
		}
	}
}

thread_local LogScope* Logger::m_scope = 0;

AllocBackEnd be;
std::atomic<int> i = 0;
std::atomic<float> k2 = 0;
void PushThread(void *arg)
{
	int loops = (int)arg;
	SimpleLogExporter exp(be, *buff_ptr);
	SimpleLogProcessor pro(std::move(exp));
	Logger logger("test", pro);
	Logger log2("log2", pro);


	for(int k=0;k<loops;k++)
	{
		AttributeValue val(GetCurrentThreadId());
		LogScope scope = logger.begin_scope({
			{"file.name", __FILE__},
			{"file.line", (int)__LINE__},
			{"thread.id", val },
			});
		logger.log(Severity::Info, "Hello World",
			{
				{"key1", (int)(i++)},
				{"key2", k2.load()},
				{"key3", "asdf" + std::to_string(k2) + " " + std::to_string(i)},
			}
		);

		k2.store(k2 + 0.001);
		{
			LogScope scop2 = log2.begin_scope({
				{"inner", "inner"},
				});
			log2.log(Severity::Info, "Nexted", {});
		}
		scope.end();
		//Sleep(100);
	}
}

void *p1, *p2, *p3;
int main()
{
	size_t st = sizeof(NamedAttribute);
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
	_beginthread(ReadThread, 0x100000, 0);
	RingBuffer buff(65536);
	buff_ptr = &buff;

	/*
	* // Check for leaks
	_CrtMemState state, state2;
	_CrtMemCheckpoint(&state);
	PushThread(1000);
	_CrtMemCheckpoint(&state2);
	_CrtMemDumpAllObjectsSince(&state);
	*/

	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);
	_beginthread(PushThread, 0, (void*)99999999);

	while (true)
	{
		Sleep(1000);
	}
}
