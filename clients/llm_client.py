"""
LLM Client - OpenRouter를 통한 비동기 LLM 호출
AsyncOpenAI를 사용한 비동기 구현
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    비동기 LLM 클라이언트 (OpenRouter 전용)
    
    OpenRouter를 통해 다양한 LLM 모델 사용
    AsyncOpenAI 클라이언트와 호환되므로 base_url만 변경
    """
    
    def __init__(
        self,
        provider: str = None,  # 호환성 유지용, 무시됨
        model: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 8192  # Claude Sonnet 4.5 최대 출력 토큰
    ):
        # 모델 설정
        if model:
            self.model = model
        else:
            self.model = os.getenv("LLM_MODEL", "anthropic/claude-4.5-sonnet")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # OpenRouter API 키
        if AsyncOpenAI is None:
            raise ImportError("openai 패키지 필요: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다. .env 파일에 추가하세요.")
        
        # OpenRouter 클라이언트 초기화
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/biocoscientist",
                "X-Title": "BioCoScientist"
            }
        )
        
        logger.info(f"LLMClient 초기화: OpenRouter - {self.model} (max_tokens={max_tokens})")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
        temperature: float = None,
        max_tokens: int = None,
        purpose: str = None,
        use_cache: bool = False,  # Prompt caching 사용 여부
        **kwargs
    ) -> str:
        """
        비동기 텍스트 생성
        
        Args:
            messages: [{"role": "user", "content": "..."}] 형식
            system: 시스템 프롬프트
            temperature: 온도 (기본값 사용시 None)
            max_tokens: 최대 토큰 수
            purpose: LLM 호출 목적 (로깅용)
            use_cache: Prompt caching 사용 (긴 컨텍스트 재사용 시 토큰 절약)
        
        Returns:
            생성된 텍스트
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # 메시지 포맷팅
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)
        
        # Prompt Caching 적용 (Claude 모델만 지원)
        if use_cache and 'claude' in self.model.lower():
            # 마지막 user 메시지 이전의 모든 메시지를 캐시
            # 이렇게 하면 debate history 같은 긴 컨텍스트를 캐시에 저장
            if len(formatted_messages) >= 2:
                # 마지막에서 두번째 메시지에 cache_control 추가
                formatted_messages[-2] = {
                    **formatted_messages[-2],
                    "cache_control": {"type": "ephemeral"}
                }
                logger.info(f"[CACHE] Prompt caching enabled for message at index {len(formatted_messages)-2}")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            
            content = response.choices[0].message.content
            purpose_str = f"purpose={purpose}" if purpose else f"temp={temp}"
            
            # 캐시 사용 정보 로그
            cache_info = ""
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'prompt_tokens_details'):
                    cached = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                    if cached > 0:
                        cache_info = f" | cached={cached} tokens (↓{int(cached/usage.prompt_tokens*100)}%)"
            
            logger.info(f"[LLM] {self.model} | {purpose_str} | response={len(content)} chars{cache_info}")
            
            return content
            
        except Exception as e:
            print(f"[DEBUG LLM] API call failed: {e}")
            logger.error(f"LLM 생성 오류: {e}")
            raise
    
    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
        purpose: str = None,
        use_cache: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        비동기 JSON 응답 생성
        
        Args:
            purpose: LLM 호출 목적 (로깅용)
            use_cache: Prompt caching 사용 (긴 debate history 등)
        
        Returns:
            파싱된 JSON 딕셔너리
        """
        
        # JSON 형식 요청 힌트 추가
        if messages:
            last_content = messages[-1].get('content', '')
            if 'json' not in last_content.lower():
                messages = messages.copy()
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": last_content + "\n\nRespond in valid JSON format."
                }
        
        response = await self.generate(messages, system, purpose=purpose, use_cache=use_cache, **kwargs)
        
        # 빈 응답 체크
        if not response or not response.strip():
            logger.error("Empty response from LLM")
            raise ValueError("LLM returned empty response")
        
        # JSON 파싱
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            # Claude가 마크다운 코드 블록으로 감쌌서 반환하는 경우가 많음 - 이건 정상
            
            # 1. 마크다운 코드블록에서 JSON 추출
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 2. 가장 큰 완전한 JSON 객체 찾기 (네스트된 구조 고려)
            # 중괄호 균형 맞추기
            start = response.find('{')
            if start != -1:
                brace_count = 0
                for i, char in enumerate(response[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            try:
                                return json.loads(response[start:start+i+1])
                            except json.JSONDecodeError:
                                pass
            
            # 3. 잘린 JSON 복구 시도 (마지막 완전한 객체/배열까지만 사용)
            start = response.find('{')
            if start != -1:
                # 마지막으로 완전히 닫힌 위치까지 찾기
                for end in range(len(response), start, -1):
                    candidate = response[start:end].rstrip()
                    # 미완성 문자열/배열 제거
                    for try_end in [candidate.rfind('}'), candidate.rfind(']')]:
                        if try_end > 0:
                            try:
                                return json.loads(candidate[:try_end+1])
                            except json.JSONDecodeError:
                                continue
            
            # 모든 파싱 시도 실패
            logger.error(f"All JSON parsing attempts failed")
            logger.error(f"Full response ({len(response)} chars): {response}")
            
            # 사용자에게 명확한 오류 메시지
            error_msg = f"JSON 파싱 실패. LLM 응답 미리보기:\n{response[:500]}\n\n전체 응답은 로그를 확인하세요."
            raise ValueError(error_msg)
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        system: str = None,
        temperature: float = None,
        max_iterations: int = 5,
        purpose: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tool calling을 지원하는 비동기 생성
        
        LLM이 tool을 호출할 수 있으며, 여러 번의 왕복이 가능합니다.
        
        Args:
            messages: 대화 메시지
            tools: Tool 정의 리스트 (OpenAI function calling 형식)
            tool_choice: "auto", "none", 또는 특정 tool 이름
            system: 시스템 프롬프트
            temperature: 온도
            max_iterations: 최대 tool call 반복 횟수
            purpose: 로깅용 목적
            
        Returns:
            {
                "content": "최종 응답 텍스트",
                "tool_calls": [{"name": "tool_name", "arguments": {...}, "result": ...}],
                "finish_reason": "stop" | "tool_calls" | "length"
            }
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = self.max_tokens
        
        # 메시지 포맷팅
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)
        
        tool_call_history = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temp,
                    max_tokens=max_tok,
                    **kwargs
                )
                
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                
                purpose_str = f"purpose={purpose}" if purpose else f"tool_calling"
                logger.info(f"[LLM] {self.model} | {purpose_str} | iteration={iteration} | finish={finish_reason}")
                
                # Tool call이 없으면 종료
                if not message.tool_calls:
                    return {
                        "content": message.content or "",
                        "tool_calls": tool_call_history,
                        "finish_reason": finish_reason
                    }
                
                # Tool call 실행 - 호출자가 처리하도록 반환
                # Agent layer handles tool execution
                for tool_call in message.tool_calls:
                    tool_info = {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "result": None  # 호출자가 채워야 함
                    }
                    tool_call_history.append(tool_info)
                
                # Tool call이 있으면 일단 반환 (호출자가 실행 후 다시 호출)
                # Agent layer will execute tools and continue conversation
                return {
                    "content": message.content or "",
                    "tool_calls": tool_call_history,
                    "finish_reason": finish_reason,
                    "pending_tool_calls": True  # 아직 tool 실행 필요
                }
                
            except Exception as e:
                logger.error(f"Tool calling error at iteration {iteration}: {e}")
                raise
        
        # Max iterations 도달
        logger.warning(f"Max iterations ({max_iterations}) reached in tool calling")
        return {
            "content": "",
            "tool_calls": tool_call_history,
            "finish_reason": "max_iterations"
        }
    
    async def continue_with_tool_results(
        self,
        messages: List[Dict[str, str]],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str = None,
        temperature: float = None,
        purpose: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tool 실행 결과를 받아서 대화를 계속 진행
        
        Args:
            messages: 기존 대화 메시지 (user message만)
            tool_calls: 이전 응답의 tool_calls [{"id": "...", "name": "...", "arguments": {...}}]
            tool_results: [{"id": "call_xxx", "name": "tool_name", "result": "..."}]
            tools: Tool 정의
            system: 시스템 프롬프트
            temperature: 온도
            purpose: 로깅용 목적
            
        Returns:
            최종 응답 또는 추가 tool call
        """
        temp = temperature if temperature is not None else self.temperature
        
        # 메시지 포맷팅
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        formatted_messages.extend(messages)
        
        # Assistant의 tool call 메시지 추가
        formatted_messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"])
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # Tool 결과를 메시지에 추가
        for tool_result in tool_results:
            formatted_messages.append({
                "role": "tool",
                "tool_call_id": tool_result["id"],
                "name": tool_result["name"],
                "content": str(tool_result["result"])
            })
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools,
                temperature=temp,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            purpose_str = f"purpose={purpose}" if purpose else "tool_continue"
            logger.info(f"[LLM] {self.model} | {purpose_str} | finish={finish_reason}")
            
            # 추가 tool call이 있는지 확인
            if message.tool_calls:
                pending_calls = []
                for tool_call in message.tool_calls:
                    pending_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                        "result": None
                    })
                
                return {
                    "content": message.content or "",
                    "tool_calls": pending_calls,
                    "finish_reason": finish_reason,
                    "pending_tool_calls": True
                }
            
            # 최종 응답
            return {
                "content": message.content or "",
                "tool_calls": [],
                "finish_reason": finish_reason,
                "pending_tool_calls": False
            }
            
        except Exception as e:
            logger.error(f"Tool continuation error: {e}")
            raise


# ============================================================================
# Embedding Client (선택적)
# ============================================================================

class EmbeddingClient:
    """임베딩 생성 클라이언트 (OpenRouter 사용)"""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-3-small"
    ):
        if AsyncOpenAI is None:
            raise ImportError("openai 패키지 필요: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY가 설정되지 않았습니다.")
        
        self.model = model
        
        # OpenRouter를 통한 임베딩 API 호출
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/biocoscientist",
                "X-Title": "BioCoScientist"
            }
        )
        
        logger.info(f"EmbeddingClient 초기화: OpenRouter - {self.model}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트 임베딩 생성"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩 생성 (배치)"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import numpy as np
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# 편의를 위한 싱글톤 인스턴스
_default_client = None

def get_client() -> LLMClient:
    """기본 LLM 클라이언트 반환"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client

