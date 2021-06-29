#include <process.h>
#include "LogRecord.h"
#include <process.h>

void test_RingBuffer()
{
	RingBuffer b(128 + 12);

	char dst[100];
	for (int i = 0; i < 100000; i++)
	{
		char* p1 = b.reserve(6);
		char* p2 = b.reserve(9);

		sprintf(p1, "a%d", i);
		sprintf(p2, "b%d", i);
		b.commit(p1);
		b.commit(p2);
		
		size_t len = b.consume(dst, 100);
		printf("%d %s %d\n", i, dst, len);
	}
}

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
		printf("%d", b);
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

	void operator()(Span<const char> b)
	{
		printf("%.*s", b.len, b.data);
	}
};
SimpleLogExporter* e;
void ReadThread(void* )
{
	while (true)
	{
		SimpleLogRecord slr;
		bool b1 = e->read_record(slr);
		if (b1)
		{
			printf("%llu ", slr.ts);
			printf("%04x ", slr.severity);
			printf("%.*s - ", slr.message.len, slr.message.data);
			for (auto it = slr.m_attrs.begin();
				it != slr.m_attrs.end();
				it++)
			{
				printf("\n\t%s : ", it->first.c_str());

				visit(AttrPrinter(), it->second);
			};
			printf("\n");
		}
	}
}
int main()
{
	_beginthread(ReadThread, 0, 0);
	RingBuffer buff(65536);
	AllocBackEnd be;
	SimpleLogExporter exp(be, buff);
	SimpleLogProcessor pro( std::move(exp) );
	e = &exp;
	/*
	LogRecord* r = pro.create_record();
	r->set_name("TestLogger");
	r->set_severity(Severity::Info);
	r->set_message("This is my messaeg");
	r->add_attribute("key", 1);
	r->add_attribute("key2", 1.1);
	r->add_attribute("key3", 5.0f);
	pro.add_record(r);
	*/

	Logger logger("test", pro);

	int i = 0;
	while (true)
	{
		LogScope scope = logger.begin_scope({
			{"file.name", __FILE__},
			{"file.line", (int)__LINE__},
			});
		logger.log(Severity::Info, "Hello World",
			{
				{"key1", i++},
				{"key2", 2.2},
				{"key3", "asdf"},
			}
		);
		//scope.end();
	}
}
