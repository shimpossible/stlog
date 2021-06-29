#pragma once
#include "ObjectValue.h"
#include "LogEncoder.h"

class LoggerScope
{
public:
	LoggerScope() {}

	LoggerScope(ObjectValue&& d)
		: data(std::move(d) )
	{
	}

	virtual ~LoggerScope()
	{
	}

	ObjectValue data;

	/**
	 Record scope to buffer
	 */
	virtual void encode(LogEncoder& e)
	{
		if (data.data.size() == 0) return;

		// number of entries
		*e.buffer++ = data.data.size();
		for (ObjectValue::Iterator it = data.data.begin();
			it != data.data.end();
			it++)
		{
			e(*it);			
		}
	}
};
