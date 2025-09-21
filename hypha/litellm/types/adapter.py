from typing import List

from typing_extensions import Dict, Required, TypedDict, override

from hypha.litellm.integrations.custom_logger import CustomLogger


class AdapterItem(TypedDict):
    id: str
    adapter: CustomLogger
