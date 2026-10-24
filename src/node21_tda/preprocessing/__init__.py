"""Preprocessing paketi: segmentasyon ve 5 anatomik bölge kırpma araçları.

NOT: `zones` (ve onun çektiği matplotlib) **TEMBEL** yüklenir. Bu sayede gönderim
konteyneri yalnız segmenteri kullanırken matplotlib'e bağımlı olmaz (PEP 562).
"""

from node21_tda.preprocessing.segmenter import PSPNet5ZoneSegmenter, ZONE_METADATA

__all__ = ["PSPNet5ZoneSegmenter", "ZoneCropper", "ZONE_METADATA"]


def __getattr__(name):
    """Tembel erişim: `ZoneCropper` istenince `zones` modülü yüklenir."""
    if name == "ZoneCropper":
        from node21_tda.preprocessing.zones import ZoneCropper
        return ZoneCropper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
