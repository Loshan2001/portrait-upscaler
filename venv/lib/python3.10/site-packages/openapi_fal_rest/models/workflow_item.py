import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="WorkflowItem")


@attr.s(auto_attribs=True)
class WorkflowItem:
    """
    Attributes:
        name (str):
        title (str):
        user_id (str):
        user_nickname (str):
        created_at (datetime.datetime):
        thumbnail_url (Union[Unset, str]):
    """

    name: str
    title: str
    user_id: str
    user_nickname: str
    created_at: datetime.datetime
    thumbnail_url: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        title = self.title
        user_id = self.user_id
        user_nickname = self.user_nickname
        created_at = self.created_at.isoformat()

        thumbnail_url = self.thumbnail_url

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "user_id": user_id,
                "user_nickname": user_nickname,
                "created_at": created_at,
            }
        )
        if thumbnail_url is not UNSET:
            field_dict["thumbnail_url"] = thumbnail_url

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        user_id = d.pop("user_id")

        user_nickname = d.pop("user_nickname")

        created_at = isoparse(d.pop("created_at"))

        thumbnail_url = d.pop("thumbnail_url", UNSET)

        workflow_item = cls(
            name=name,
            title=title,
            user_id=user_id,
            user_nickname=user_nickname,
            created_at=created_at,
            thumbnail_url=thumbnail_url,
        )

        workflow_item.additional_properties = d
        return workflow_item

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
