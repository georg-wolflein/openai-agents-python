import copy
import json
from typing import Any, Literal

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypeAliasType

from agents import Agent, AgentOutputSchema, RunConfig, Runner, function_tool
from agents.exceptions import UserError
from agents.models.openai_responses import OpenAIResponsesModel
from agents.strict_schema import ensure_strict_json_schema
from tests.model_test_helpers import get_response_obj
from tests.test_responses import get_function_tool_call, get_text_message

AnyJsonValue = TypeAliasType(
    "AnyJsonValue",
    str | int | float | bool | list["AnyJsonValue"] | dict[str, "AnyJsonValue"] | None,
)


class Result(BaseModel):
    status: Literal["success", "failure"]
    metadata: AnyJsonValue


# An independent wire-schema oracle, not recomputed by the normalizer under test.
_JSON_SCHEMA = {
    "$defs": {
        "AnyJsonValue": {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "number"},
                {"type": "boolean"},
                {"type": "array", "items": {"$ref": "#/$defs/AnyJsonValue"}},
                {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/AnyJsonValue"},
                    "properties": {},
                    "required": [],
                },
                {"type": "null"},
            ]
        }
    },
    "type": "object",
    "title": "Result",
    "properties": {
        "status": {"title": "Status", "type": "string", "enum": ["success", "failure"]},
        "metadata": {"$ref": "#/$defs/AnyJsonValue"},
    },
    "required": ["status", "metadata"],
    "additionalProperties": False,
}
_VALUE = {"nested": [{"count": 3, "optional": None}, 2.5, True, ["x"]]}
_RESULT = {"status": "success", "metadata": _VALUE}


def test_recursive_json_output_preserves_strict_outer_schema_and_is_idempotent():
    output = AgentOutputSchema(Result, strict_json_schema=True)

    assert output.is_strict_json_schema()
    assert output.json_schema() == _JSON_SCHEMA
    assert ensure_strict_json_schema(copy.deepcopy(output.json_schema())) == _JSON_SCHEMA
    assert output.validate_json(json.dumps(_RESULT)) == Result(status="success", metadata=_VALUE)


def test_recursive_json_field_description_expands_reference():
    class DescribedOutput(Result):
        metadata: AnyJsonValue = Field(description="Any JSON value.")

    output = AgentOutputSchema(DescribedOutput)

    assert output.json_schema() == {
        **_JSON_SCHEMA,
        "title": "DescribedOutput",
        "properties": {
            **_JSON_SCHEMA["properties"],
            "metadata": {
                **_JSON_SCHEMA["$defs"]["AnyJsonValue"],
                "description": "Any JSON value.",
            },
        },
    }


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["output", "tool"])
async def test_recursive_json_through_runner_and_responses(mode):
    received: list[Any] = []

    @function_tool
    def store_value(status: Literal["success", "failure"], metadata: AnyJsonValue) -> str:
        received.append({"status": status, "metadata": metadata})
        return "stored"

    # The provider request body is part of the regression, so use an HTTP mock rather
    # than a provider-neutral ScriptedModel. All response data below is synthetic.
    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if mode == "output":
            assert body["text"]["format"]["strict"] is True
            assert body["text"]["format"]["schema"] == _JSON_SCHEMA
            item = get_text_message(json.dumps(_RESULT))
        else:
            assert body["tools"][0]["strict"] is True
            assert body["tools"][0]["parameters"] == {
                **_JSON_SCHEMA,
                "title": "store_value_args",
            }
            item = get_function_tool_call("store_value", json.dumps(_RESULT))
        return httpx2.Response(200, json=get_response_obj([item]).model_dump())

    async with AsyncOpenAI(
        api_key="test-key",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        agent = Agent(
            name="typed-json",
            model=OpenAIResponsesModel(model="test-model", openai_client=client),
            output_type=AgentOutputSchema(Result, strict_json_schema=True)
            if mode == "output"
            else None,
            tools=[store_value] if mode == "tool" else [],
            tool_use_behavior="stop_on_first_tool",
        )
        result = await Runner.run(
            agent, "Synthetic JSON probe", run_config=RunConfig(tracing_disabled=True)
        )

    if mode == "output":
        assert result.final_output == Result(status="success", metadata=_VALUE)
        assert received == []
    else:
        assert result.final_output == "stored"
        assert received == [_RESULT]


def test_pydantic_integer_map_gets_explicit_empty_properties():
    class MapOutput(BaseModel):
        value: dict[str, int]

    schema = AgentOutputSchema(MapOutput).json_schema()

    assert schema["properties"]["value"] == {
        "title": "Value",
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": {"type": "integer"},
    }


def test_typed_map_values_use_existing_strict_normalization():
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "object", "properties": {"count": {"type": "integer"}}},
                    {"type": "null", "default": None},
                ]
            },
        },
    }

    assert ensure_strict_json_schema(schema) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
        },
    }


@pytest.mark.parametrize("value_schema", [True, {}, {"description": "Any value"}])
def test_untyped_map_values_remain_rejected(value_schema):
    with pytest.raises(UserError, match="additionalProperties must be false or a typed schema"):
        ensure_strict_json_schema({"type": "object", "additionalProperties": value_schema})


def test_typed_map_cannot_hide_unrestricted_nested_objects():
    with pytest.raises(UserError, match="Unrestricted additional properties"):
        ensure_strict_json_schema(
            {
                "type": "object",
                "additionalProperties": {"type": "object", "additionalProperties": True},
            }
        )


def test_typed_map_values_share_the_conversion_node_budget(monkeypatch):
    monkeypatch.setattr("agents.strict_schema._MAX_SCHEMA_NODES", 2)

    with pytest.raises(UserError, match="too large"):
        ensure_strict_json_schema(
            {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "integer"}},
            }
        )


def test_typed_map_ref_annotation_is_expanded():
    schema = {
        "$defs": {"Item": {"type": "integer"}},
        "type": "object",
        "additionalProperties": {"$ref": "#/$defs/Item", "description": "Count"},
    }

    assert ensure_strict_json_schema(schema)["additionalProperties"] == {
        "type": "integer",
        "description": "Count",
    }


def test_typed_map_ref_constraints_are_not_discarded():
    schema = {
        "$defs": {"Item": {"type": "integer"}},
        "type": "object",
        "additionalProperties": {"$ref": "#/$defs/Item", "minimum": 1},
    }

    with pytest.raises(UserError, match="incompatible sibling"):
        ensure_strict_json_schema(schema)


def test_typed_map_ref_inside_nested_resource_is_rejected():
    schema = {
        "$defs": {"Item": {"type": "integer"}},
        "type": "object",
        "properties": {
            "map": {
                "$id": "https://example.test/nested",
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/Item"},
            }
        },
    }

    with pytest.raises(UserError, match=r"nested `\$id`"):
        ensure_strict_json_schema(schema)


def test_singleton_allof_preserves_typed_value_constraints():
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "allOf": [{"description": "Counts by name"}],
    }

    assert ensure_strict_json_schema(schema) == {
        "type": "object",
        "description": "Counts by name",
        "additionalProperties": {"type": "integer"},
        "properties": {},
        "required": [],
    }


def test_singleton_allof_cannot_exempt_fields_from_parent_map_value_constraints():
    schema = {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "allOf": [{"properties": {"name": {"type": "string"}}}],
    }

    with pytest.raises(UserError, match="singleton `allOf`"):
        ensure_strict_json_schema(schema)
