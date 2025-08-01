---
search:
  exclude: true
---
# ツール

ツールを使用すると エージェント がアクションを実行できます。データの取得、コードの実行、外部 API の呼び出し、さらには コンピュータ操作 などが可能です。Agents SDK には次の 3 つの カテゴリー のツールがあります。

- ホスト型ツール: これらは LLM サーバー上で AI モデルと並行して実行されます。OpenAI は retrieval、Web 検索、コンピュータ操作 をホスト型ツールとして提供しています。
- 関数呼び出し (Function Calling): 任意の Python 関数をツールとして利用できます。
- ツールとしてのエージェント: エージェント をツールとして使用でき、ハンドオフ せずに他の エージェント を呼び出せます。

## ホスト型ツール

[`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel] を使用する場合、OpenAI はいくつかの組み込みツールを提供しています。

- [`WebSearchTool`][agents.tool.WebSearchTool] を使用すると エージェント が Web 検索を実行できます。
- [`FileSearchTool`][agents.tool.FileSearchTool] は OpenAI ベクトルストア から情報を取得します。
- [`ComputerTool`][agents.tool.ComputerTool] は コンピュータ操作 タスクを自動化します。
- [`CodeInterpreterTool`][agents.tool.CodeInterpreterTool] は LLM がサンドボックス環境でコードを実行できるようにします。
- [`HostedMCPTool`][agents.tool.HostedMCPTool] は リモート MCP サーバー のツールをモデルに公開します。
- [`ImageGenerationTool`][agents.tool.ImageGenerationTool] はプロンプトから画像を生成します。
- [`LocalShellTool`][agents.tool.LocalShellTool] はローカルマシンでシェルコマンドを実行します。

```python
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
        ),
    ],
)

async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
    print(result.final_output)
```

## 関数ツール

任意の Python 関数をツールとして使用できます。Agents SDK がツールの設定を自動で行います。

- ツール名は Python 関数名になります (別名を指定することも可能です)。
- ツールの説明は関数の docstring から取得されます (説明を直接指定することも可能です)。
- 関数の引数から入力スキーマが自動生成されます。
- 各入力の説明は docstring から取得されます (無効化も可能です)。

Python の `inspect` モジュールを使用して関数シグネチャを抽出し、[`griffe`](https://mkdocstrings.github.io/griffe/) で docstring を解析し、スキーマの作成には `pydantic` を利用しています。

```python
import json

from typing_extensions import TypedDict, Any

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


class Location(TypedDict):
    lat: float
    long: float

@function_tool  # (1)!
async def fetch_weather(location: Location) -> str:
    # (2)!
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny"


@function_tool(name_override="fetch_data")  # (3)!
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents>"


agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file],  # (4)!
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()

```

1. 引数には任意の Python 型を使用でき、関数は同期・非同期のいずれでも構いません。  
2. docstring がある場合、全体の説明および引数の説明を取得します。  
3. 関数はオプションで `context` を受け取れます (先頭の引数である必要があります)。また、ツール名や説明、docstring スタイルなどを上書き設定できます。  
4. 装飾された関数をツール一覧に渡してください。

??? note "クリックして出力を表示"

    ```
    fetch_weather
    Fetch the weather for a given location.
    {
    "$defs": {
      "Location": {
        "properties": {
          "lat": {
            "title": "Lat",
            "type": "number"
          },
          "long": {
            "title": "Long",
            "type": "number"
          }
        },
        "required": [
          "lat",
          "long"
        ],
        "title": "Location",
        "type": "object"
      }
    },
    "properties": {
      "location": {
        "$ref": "#/$defs/Location",
        "description": "The location to fetch the weather for."
      }
    },
    "required": [
      "location"
    ],
    "title": "fetch_weather_args",
    "type": "object"
    }

    fetch_data
    Read the contents of a file.
    {
    "properties": {
      "path": {
        "description": "The path to the file to read.",
        "title": "Path",
        "type": "string"
      },
      "directory": {
        "anyOf": [
          {
            "type": "string"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "description": "The directory to read the file from.",
        "title": "Directory"
      }
    },
    "required": [
      "path"
    ],
    "title": "fetch_data_args",
    "type": "object"
    }
    ```

### カスタム関数ツール

Python 関数をツールとして使いたくない場合は、直接 [`FunctionTool`][agents.tool.FunctionTool] を作成できます。必要な項目は以下です。

- `name`
- `description`
- `params_json_schema` : 引数の JSON スキーマ
- `on_invoke_tool` : [`ToolContext`][agents.tool_context.ToolContext] と引数 (JSON 文字列) を受け取り、ツールの出力文字列を返す非同期関数

```python
from typing import Any

from pydantic import BaseModel

from agents import RunContextWrapper, FunctionTool



def do_some_work(data: str) -> str:
    return "done"


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)
```

### 引数と docstring の自動解析

前述のとおり、関数シグネチャを自動解析してツールのスキーマを抽出し、docstring を解析してツールおよび個々の引数の説明を取得します。主なポイントは以下のとおりです。

1. シグネチャの解析は `inspect` モジュールで行います。型アノテーションを用いて引数の型を理解し、Pydantic モデルを動的に生成します。Python の基本型、Pydantic モデル、TypedDict など大半の型をサポートします。  
2. docstring の解析には `griffe` を使用します。サポートされる docstring 形式は `google`、`sphinx`、`numpy` です。形式はベストエフォートで自動判定しますが、`function_tool` 呼び出し時に明示的に指定することもできます。`use_docstring_info` を `False` にすると docstring 解析を無効化できます。

スキーマ抽出のコードは [`agents.function_schema`][] にあります。

## ツールとしてのエージェント

一部のワークフローでは、ハンドオフ する代わりに中央の エージェント が複数の専門 エージェント をオーケストレーションしたい場合があります。そのようなときは エージェント をツールとしてモデル化できます。

```python
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

async def main():
    result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)
```

### ツールエージェントのカスタマイズ

`agent.as_tool` は エージェント を簡単にツール化するための便利メソッドです。ただしすべての設定をサポートしているわけではありません。たとえば `max_turns` は設定できません。高度なユースケースでは、ツール実装内で `Runner.run` を直接呼び出してください。

```python
@function_tool
async def run_my_agent() -> str:
    """A tool that runs the agent with custom configs"""

    agent = Agent(name="My agent", instructions="...")

    result = await Runner.run(
        agent,
        input="...",
        max_turns=5,
        run_config=...
    )

    return str(result.final_output)
```

### 出力抽出のカスタマイズ

場合によっては、ツールエージェント の出力を中央 エージェント に返す前に加工したいことがあります。たとえば以下のようなケースです。

- サブエージェントのチャット履歴から特定情報 (例: JSON ペイロード) を抽出したい。  
- エージェント の最終回答を変換・再整形したい (例: Markdown をプレーンテキストや CSV に変換)。  
- エージェント の応答が欠落している、または不正な場合に検証やフォールバック値を提供したい。  

そのような場合は、`as_tool` メソッドに `custom_output_extractor` 引数を渡してください。

```python
async def extract_json_payload(run_result: RunResult) -> str:
    # Scan the agent’s outputs in reverse order until we find a JSON-like message from a tool call.
    for item in reversed(run_result.new_items):
        if isinstance(item, ToolCallOutputItem) and item.output.strip().startswith("{"):
            return item.output.strip()
    # Fallback to an empty JSON object if nothing was found
    return "{}"


json_tool = data_agent.as_tool(
    tool_name="get_data_json",
    tool_description="Run the data agent and return only its JSON payload",
    custom_output_extractor=extract_json_payload,
)
```

## 関数ツールでのエラー処理

`@function_tool` で関数ツールを作成する際、`failure_error_function` を渡すことができます。これはツール呼び出しがクラッシュした場合に LLM へ返すエラーレスポンスを生成する関数です。

- 何も渡さない場合、デフォルトで `default_tool_error_function` が実行され、LLM にエラーが発生したことを伝えます。  
- 独自のエラー関数を渡した場合、その関数が実行され、その結果が LLM に送信されます。  
- 明示的に `None` を渡すと、ツール呼び出し時のエラーは再スローされます。たとえばモデルが無効な JSON を生成した場合は `ModelBehaviorError`、ユーザーのコードがクラッシュした場合は `UserError` などになります。  

`FunctionTool` オブジェクトを手動で作成する場合は、`on_invoke_tool` 関数内でエラーを処理する必要があります。