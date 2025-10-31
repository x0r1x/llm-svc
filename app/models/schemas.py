
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum

class FunctionCall(BaseModel):
    name: str = Field(..., description="Название вызываемой функции")
    arguments: str = Field(..., description="Аргументы функции в формате JSON строки")

class ToolCall(BaseModel):
    id: str = Field(..., description="Уникальный идентификатор вызова инструмента")
    type: Literal["function"] = "function"
    function: FunctionCall

class FunctionDefinition(BaseModel):
    name: str = Field(..., description="Название функции")
    description: Optional[str] = Field(None, description="Описание функции")
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema параметров функции")

class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class Message(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    name: Optional[str] = None

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
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None

class ModelsListResponse(BaseModel):
    data: List[Dict[str, Any]]
    object: str = "list"
