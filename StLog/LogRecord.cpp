#include "LogRecord.h"

LogScope Logger::begin_scope(std::initializer_list<NamedAttribute> at)
{
	LogProcessor* pro = m_provider.get_processor();
	AttributeList* attrs = pro->create_attribute_list();

	for (const NamedAttribute* it = at.begin();
		it < at.end();
		it++)
	{
		attrs->add_attribute(*it);
	}
	LogScope scope(attrs);
	scope.logger = this;
	scope.next = m_scope;
	m_scope = &scope;
	return scope;
}

void Logger::log(Severity sev, const char* msg, std::initializer_list<NamedAttribute> attrs)
{
	LogProcessor* pro = m_provider.get_processor();
	LogRecord* r = pro->create_record();
	uint64_t nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()
		).count();

	r->set_timestamp(nsec);
	r->set_name(m_name);
	r->set_message(msg);
	r->set_severity(sev);

	const NamedAttribute* it;

	// Add scope values
	LogScope* curr = m_scope;
	int depth = 0;
	while (curr)
	{
		for (it = curr->attrs->begin(); it < curr->attrs->end(); it++)
		{
			r->add_attribute(it->name, it->value);
		}
		curr = curr->next;
	}

	// add local values
	for (it = attrs.begin(); it < attrs.end(); it++)
	{
		r->add_attribute(it->name, it->value);
	}

	pro->add_record(r);
}

void LogScope::end()
{
	if (logger == 0) return;

	LogProcessor* pro = logger->m_provider.get_processor();
	pro->release_attribute_list(attrs);
	logger->m_scope = next;
	logger = 0;
}

LogScope::LogScope(LogScope&& other) noexcept
{
	logger = std::move(other.logger);
	attrs = std::move(other.attrs);
	next = std::move(other.next);

	assert(logger->m_scope == &other);
	logger->m_scope = this;

	other.logger = nullptr;
	other.next = nullptr;
	other.attrs = nullptr;
}
LogScope::~LogScope()
{
	// implicit end of scope
	end();
}
