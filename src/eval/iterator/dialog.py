# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from abc import abstractmethod, ABC
from logging import getLogger
from pathlib import Path
from typing import (
    List,
    Any,
    Union,
    Literal,
    Collection,
    AbstractSet,
    Optional,
    Sequence,
    Type,
)

logger = getLogger()

import typing as tp
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MessageBase(ABC):
    # Source is what we use to call "role". e.g. user/assistant/system/ipython
    source: str

    version: str

    # Primary contents of the message
    body: tp.Optional[str] = None

    # Destination is the target for the message (e.g. one of the valid sources)
    # In general if the message target is non-obvious None is a good default
    # E.g. for system message no obvious destination so use the default of None
    destination: tp.Optional[str] = None

    # We want the model to be able to signal when it wants to hand control back to the user.
    eot: bool = False

    # Use metadata for experimental information, if it stabilizes we can include it in the next version of this datatype
    metadata: tp.Optional[tp.Any] = None

    # Whether it is an ipython call, this is for MessageV2 only
    ipython: bool = False

    @classmethod
    @abstractmethod
    def from_dict(cls, repr: tp.Dict) -> "MessageBase":
        ...

    @abstractmethod
    def assert_valid(self) -> None:
        ...

    @classmethod
    @abstractmethod
    def system(cls, body: tp.Optional[str] = None) -> "MessageBase":
        ...

    @classmethod
    @abstractmethod
    def user(cls, body: tp.Optional[str] = None) -> "MessageBase":
        ...

    @classmethod
    @abstractmethod
    def assistant(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "MessageBase":
        ...

    @classmethod
    @abstractmethod
    def ipython_call(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "MessageBase":
        ...

    @classmethod
    @abstractmethod
    def ipython_return(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "MessageBase":
        ...


@dataclass
class Message(MessageBase):

    # This should be left as the default, it can be used to keep track of messages on disk that use a different type/version
    version: str = "message_v1"

    @classmethod
    def system(cls, body: tp.Optional[str] = None) -> "Message":
        return Message(source="system", body=body)

    # Helper builder functions to illustrate how to use this message type
    @classmethod
    def user(cls, body: tp.Optional[str] = None) -> "Message":
        return Message(source="user", body=body)

    @classmethod
    def assistant(cls, body: tp.Optional[str] = None, eot: bool = False) -> "Message":
        return Message(source="assistant", destination="user", eot=eot, body=body)

    @classmethod
    def ipython_call(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "Message":
        return Message(source="assistant", destination="ipython", eot=eot, body=body)

    @classmethod
    def ipython_return(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "Message":
        return Message(source="ipython", body=body)

    @classmethod
    def ipython_markdown_call(cls, body: tp.Optional[str] = None) -> "Message":
        return Message(source="assistant", destination="ipython_markdown", body=body)

    @classmethod
    def from_dict(cls, repr: tp.Dict) -> "Message":
        return Message(
            source=repr["source"],
            destination=repr["destination"],
            eot=repr["eot"],
            body=repr["body"],
            metadata=repr.get("metadata", None),
        )

    def __post_init__(self):
        # EOT message should always be without body and Destination None
        if self.eot is True:
            assert self.body is None
            self.destination = None
        # User messages should always have Destination None
        if self.source == "user":
            self.destination = None

    def __str__(self) -> str:
        body = repr(self.body)
        dst = repr(self.destination)
        return f"[{self.source}->{dst}{',eot' if self.eot else ''}] {body}"

    def assert_valid(self):
        assert self.version == "message_v1"
        assert self.source in ["user", "assistant", "ipython", "system"]
        if self.destination is not None:
            assert self.destination in [
                "user",
                "assistant",
                "ipython",
                "ipython_markdown",
            ]

    def serialize_header(self) -> str:
        # Required fields
        lines = [f"Source: {self.source.strip()}"]
        # Optional fields
        if self.destination is not None:
            lines.append(f"Destination: {self.destination.strip()}")
        if self.eot:
            lines.append(f"EOT: {str(self.eot).lower()}")
        return "\n".join(lines)

    def serialize_body(self) -> str:
        if self.body is None:
            return ""
        return self.body.strip()


@dataclass
class MessageV2(MessageBase):
    # This should be left as the default, it can be used to keep track of messages on disk that use a different type/version
    version: str = "message_v2"

    # Helper builder functions to illustrate how to use this message type.
    @classmethod
    def system(cls, body: tp.Optional[str] = None) -> "MessageV2":
        return MessageV2(source="system", body=body, eot=True)

    @classmethod
    def user(cls, body: tp.Optional[str] = None) -> "MessageV2":
        return MessageV2(source="user", body=body, eot=True)

    @classmethod
    def assistant(cls, body: tp.Optional[str] = None, eot: bool = False) -> "MessageV2":
        return MessageV2(source="assistant", body=body, eot=eot)

    @classmethod
    def assistant_eot(cls, body: tp.Optional[str] = None) -> "MessageV2":
        return MessageV2(source="assistant", body=body, eot=True)

    @classmethod
    def ipython_call(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "MessageV2":
        return MessageV2(source="assistant", body=body, eot=eot, ipython=True)

    @classmethod
    def ipython_return(
        cls, body: tp.Optional[str] = None, eot: bool = False
    ) -> "MessageV2":
        return MessageV2(source="ipython", body=body, eot=eot)

    @classmethod
    def from_dict(cls, repr: tp.Dict) -> "MessageV2":
        return MessageV2(
            source=repr["source"],
            eot=repr["eot"],
            body=repr["body"],
            metadata=repr["metadata"],
            ipython=repr["ipython"],
        )

    def __str__(self) -> str:
        body = repr(self.body)
        ending = "eot" if self.eot else "eom"
        return f"[{self.source},{ending}] {body}"

    def assert_valid(self) -> None:
        self.check_version()
        self.check_source()
        self.check_destination()
        self.check_body()
        self.check_eot()

    def check_source(self) -> None:
        assert self.source in ["user", "assistant", "ipython", "system"]

    def check_destination(self) -> None:
        # For MessageV2 we don't use destination
        assert self.destination is None

    def check_version(self) -> None:
        assert self.version == "message_v2"

    def check_body(self) -> None:
        if self.source in ["system", "ipython"]:
            return
        assert self.body is not None and self.body.strip() != ""

    def check_eot(self) -> None:
        if self.source in ["user", "system"]:
            assert self.eot


@dataclass
class SampleSFT:
    dialog: tp.Sequence[MessageBase]

    # this is used during SFT to enforce the loss only on specific turns
    keep_loss: tp.Optional[tp.List[bool]] = None

    # Use metadata for experimental information, if it stabilizes we can include it in the next version of this datatype
    metadata: tp.Optional[tp.Any] = None

    # This should be left as the default, it can be used to keep track of messages on disk that use a different type/version
    version: str = "sample_sft_v1"

    @classmethod
    def from_dict(cls, sample: tp.Dict[str, tp.Any]) -> "SampleSFT":
        message_version = sample["dialog"][0]["version"]
        assert message_version in {"message_v1", "message_v2"}
        dialog: tp.Sequence[MessageBase]
        if message_version == "message_v1":
            dialog = [Message.from_dict(msg_dict) for msg_dict in sample["dialog"]]
        else:
            dialog = [MessageV2.from_dict(msg_dict) for msg_dict in sample["dialog"]]
        return SampleSFT(
            dialog=dialog,
            keep_loss=sample.get("keep_loss", None),
            metadata=sample.get("metadata", None),
        )

    def assert_valid(self):
        assert self.version == "sample_sft_v1"
        if self.keep_loss is not None:
            assert len(self.dialog) == len(self.keep_loss)
        for msg in self.dialog:
            msg.assert_valid()


def convert_dialog_message_v1_to_message_v2(
    dialog: tp.Sequence[Message],
) -> tp.Sequence[MessageV2]:
    converted_dialog = []
    for message in dialog:
        if message.source == "system":
            converted_dialog.append(MessageV2.system(body=message.body))
        elif message.source == "user":
            converted_dialog.append(MessageV2.user(body=message.body))
        elif message.source == "assistant" and message.destination == "ipython":
            converted_dialog.append(
                MessageV2.ipython_call(body=message.body, eot=message.eot)
            )
        elif message.source == "assistant" and message.destination == "user":
            converted_dialog.append(MessageV2.assistant(body=message.body))
        elif message.source == "ipython":
            converted_dialog.append(MessageV2.ipython_return(body=message.body))
        elif message.source == "assistant" and message.eot and message.body is None:
            converted_dialog[-1].eot = True
        else:
            raise ValueError(f"{message} is not defined")
    return converted_dialog