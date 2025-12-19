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
        
        # JSON 형식 요청 힌트 추가 (더 강력하게)
        if messages:
            last_content = messages[-1].get('content', '')
            if 'json' not in last_content.lower():
                messages = messages.copy()
                messages[-1] = {
                    "role": messages[-1]["role"],
                    "content": last_content + "\n\nIMPORTANT: Respond with ONLY valid JSON. Do NOT wrap in markdown code blocks (no ```json). Return raw JSON object directly."
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
            # Claude가 마크다운 코드 블록으로 감싸서 반환하는 경우가 많음 - 이건 정상
            logger.debug(f"Direct JSON parse failed, trying extraction methods...")

            # 1. 마크다운 코드블록에서 추출 (닫는 ``` 없어도 작동)
            # ```json 또는 ``` 로 시작하는 경우 그 이후 내용 추출
            code_block_start = re.search(r'```(?:json)?\s*\n?', response)
            if code_block_start:
                after_block = response[code_block_start.end():]
                # 닫는 ``` 찾기 (있으면 그 전까지, 없으면 전체)
                closing = after_block.find('```')
                if closing != -1:
                    block_content = after_block[:closing].strip()
                else:
                    block_content = after_block.strip()

                try:
                    return json.loads(block_content)
                except json.JSONDecodeError as block_err:
                    logger.debug(f"Code block extraction failed: {str(block_err)[:200]}")
                    # 코드블록 내용도 균형 맞추기 시도
                    balanced = self._extract_balanced_json(block_content)
                    if balanced:
                        try:
                            return json.loads(balanced)
                        except json.JSONDecodeError:
                            logger.debug("Balanced extraction from code block failed, trying brace balancing on full response...")

            # 2. 가장 큰 완전한 JSON 객체 찾기 (중괄호 균형 맞추기)
            # 첫 번째 { 찾기
            start = response.find('{')
            if start != -1:
                json_str = self._extract_balanced_json(response[start:])
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.debug("Balanced JSON extraction failed, trying truncated JSON repair...")

            # 3. 잘린 JSON 복구 시도 - 마지막 완전한 항목까지
            start = response.find('{')
            if start != -1:
                json_str = self._repair_truncated_json(response[start:])
                if json_str:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

            # 모든 파싱 시도 실패 - 응답을 파일로 저장
            logger.error(f"All JSON parsing attempts failed")

            # 실패한 응답을 파일로 저장 (디버깅용)
            from pathlib import Path
            from datetime import datetime
            debug_dir = Path("./logs/json_parse_errors")
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = debug_dir / f"failed_response_{timestamp}.txt"

            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"=== JSON Parsing Failed ===\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Response length: {len(response)} chars\n")
                f.write(f"Purpose: {purpose or 'unknown'}\n")
                f.write(f"\n=== Full Response ===\n")
                f.write(response)

            logger.error(f"Full response ({len(response)} chars) saved to: {error_file}")
            logger.error(f"Preview: {response[:500]}...")

            # 사용자에게 명확한 오류 메시지
            error_msg = f"JSON 파싱 실패. 전체 응답이 {error_file}에 저장되었습니다.\n\n미리보기:\n{response[:500]}\n..."
            raise ValueError(error_msg)

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """중괄호 균형을 맞춰 완전한 JSON 객체 추출"""
        if not text or text[0] != '{':
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]

        return None  # 균형이 안 맞음

    def _repair_truncated_json(self, text: str) -> Optional[str]:
        """잘린 JSON 복구 - hypotheses 배열의 완전한 항목까지만 추출"""
        if not text or text[0] != '{':
            return None

        # hypotheses 배열 내 마지막 완전한 객체 찾기
        # 패턴: "hypotheses": [...{...}...]
        hyp_match = re.search(r'"hypotheses"\s*:\s*\[', text)
        if hyp_match:
            array_start = hyp_match.end()
            # 마지막 완전한 } 찾기 (배열 내 객체)
            last_complete_obj = -1
            brace_count = 0
            in_string = False
            escape_next = False
            obj_start = -1

            for i, char in enumerate(text[array_start:], array_start):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if char == '{':
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and obj_start != -1:
                        last_complete_obj = i
                elif char == ']' and brace_count == 0:
                    # 배열 끝
                    break

            if last_complete_obj > 0:
                # 마지막 완전한 객체까지 + 배열 닫기 + 객체 닫기
                truncated = text[:last_complete_obj+1]
                # 배열과 객체 닫기
                repaired = truncated + ']}'
                return repaired

        return None
    
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
                    # Skip invalid tool calls
                    if not tool_call or not hasattr(tool_call, 'function') or not tool_call.function:
                        logger.warning(f"Skipping invalid tool_call: {tool_call}")
                        continue

                    # Handle None or empty arguments
                    args_str = getattr(tool_call.function, 'arguments', None)
                    if args_str:
                        try:
                            arguments = json.loads(args_str)
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse tool arguments: {e}, raw: {str(args_str)[:100]}")
                            arguments = {}
                    else:
                        arguments = {}

                    tool_info = {
                        "id": getattr(tool_call, 'id', None),
                        "name": getattr(tool_call.function, 'name', 'unknown'),
                        "arguments": arguments,
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


# 편의를 위한 싱글톤 인스턴스
_default_client = None

def get_client() -> LLMClient:
    """기본 LLM 클라이언트 반환"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client

