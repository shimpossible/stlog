#include "Logger.h"
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

SimpleLogExporter* e;
void ReadThread(void* )
{
	while (true)
	{
		SimpleLogRecord slr;
		bool b1 = e->read_record(slr);
		if (b1)
		{
			printf("%.*s - ", slr.message.len, slr.message.data);
			for (auto it = slr.m_attrs.begin();
				it != slr.m_attrs.end();
				it++)
			{
				printf("\n\t%s :", it->first.c_str());
				switch (it->second.data_type)
				{
				case AttributeType::type_s32: printf("%d", it->second.data.s32); break;
				case AttributeType::type_string: printf("%.*s", it->second.data.s.len, it->second.data.s.data); break;
				}
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
