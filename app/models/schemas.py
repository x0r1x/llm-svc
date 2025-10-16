from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum

class FunctionDefinition(BaseModel):
    """Определение отдельной функции."""
    name: str = Field(..., description="Название функции")
    description: Optional[str] = Field(None, description="Описание функции")
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema параметров функции")

class FunctionCall(BaseModel):
    """Вызов конкретной функции с аргументами."""
    name: str = Field(..., description="Название вызываемой функции")
    arguments: str = Field(..., description="Аргументы функции в формате JSON строки")

class ToolCall(BaseModel):
    """Вызов инструмента (обертка для function call)."""
    id: str = Field(..., description="Уникальный идентификатор вызова инструмента")
    type: Literal["function"] = "function"
    function: FunctionCall

class ToolDefinition(BaseModel):
    """Определение инструмента (в OpenAI API инструментом является функция)."""
    type: Literal["function"] = "function"
    function: FunctionDefinition

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str

class ChatCompletionRequestSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str

class ChatCompletionRequestUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequestAssistantMessage(BaseModel):
    # Переопределяем существующий класс, добавляя tool_calls
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None  # Заменяем старый тип на новый

class ChatCompletionRequestToolMessage(BaseModel):
    # Сообщение с результатом выполнения инструмента
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str

class ChatCompletionRequestFunctionMessage(BaseModel):
    role: Literal["function"] = "function"
    content: str

class ChatCompletionRequest(BaseModel):
    # Расширяем модель запроса для поддержки инструментов
    model: str = "abstract-model"
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(256, ge=1)
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[Literal["none", "auto"], Dict[str, Any]]] = None

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"
    delta: Optional[Dict[str, Any]] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None

class ModelsListResponse(BaseModel):
    data: List[Dict[str, Any]]
    object: str = "list"
