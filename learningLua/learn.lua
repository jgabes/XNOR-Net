-- This is my frist demo
local M = {}

local function sayMyName()
  print('Jamie')
end

function M.sayHello()
  print('Hello')
  sayMyName()
end

print('All code loaded!')

return M
