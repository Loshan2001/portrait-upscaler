import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.workflow_detail_contents import WorkflowDetailContents


T = TypeVar("T", bound="WorkflowDetail")


@attr.s(auto_attribs=True)
class WorkflowDetail:
    """
    Attributes:
        name (str):
        title (str):
        contents (WorkflowDetailContents):
        is_public (bool):
        user_id (str):
        user_nickname (str):
        created_at (datetime.datetime):
    """

    name: str
    title: str
    contents: "WorkflowDetailContents"
    is_public: bool
    user_id: str
    user_nickname: str
    created_at: datetime.datetime
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        title = self.title
        contents = self.contents.to_dict()

        is_public = self.is_public
        user_id = self.user_id
        user_nickname = self.user_nickname
        created_at = self.created_at.isoformat()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "contents": contents,
                "is_public": is_public,
                "user_id": user_id,
                "user_nickname": user_nickname,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_detail_contents import WorkflowDetailContents

        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        contents = WorkflowDetailContents.from_dict(d.pop("contents"))

        is_public = d.pop("is_public")

        user_id = d.pop("user_id")

        user_nickname = d.pop("user_nickname")

        created_at = isoparse(d.pop("created_at"))

        workflow_detail = cls(
            name=name,
            title=title,
            contents=contents,
            is_public=is_public,
            user_id=user_id,
            user_nickname=user_nickname,
            created_at=created_at,
        )

        workflow_detail.additional_properties = d
        return workflow_detail

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
