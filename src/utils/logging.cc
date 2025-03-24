/*
 * Author: mingjian (lory.gjh@alibaba-inc.com)
 * Created Date: 2025-02-17 17:57:41
 * Modified By: mingjian (lory.gjh@alibaba-inc.com)
 * Last Modified: 2025-02-18 10:58:09
 * -----
 * Copyright (c) 2025 Alibaba Inc.
 */

#include "logging.h"

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>

void InitLogging(bool verbose)
{
  static bool initialized = false;
  if (initialized)
    return;

  static plog::ColorConsoleAppender<plog::TxtFormatter> console_appender;
  plog::init(plog::info, &console_appender);

  if (verbose)
    plog::get()->setMaxSeverity(plog::verbose);

  initialized = true;
}