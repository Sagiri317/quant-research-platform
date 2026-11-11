"""Catalog labels explain formulas while keeping persisted identifiers stable."""

import re

from quant_research_platform.interfaces.queries import list_factors
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.factors.names import factor_display_name


def test_all_public_factor_names_preserve_machine_identities():
    for profile in ("a_share_csmar_choice", "a_share_level2"):
        for factor in research_profile(profile).factors.list():
            assert re.search(r"[\u4e00-\u9fff]", factor.display_name)
    public = {item.factor_id: item for item in list_factors(research_profile()).items}
    assert public["price_volume_alpha101_054"].output_column == "alpha101_054"
    assert public["price_volume_alpha191_046"].output_column == "alpha191_046"
    assert public["cn_redemption_stock_pressure"].display_name == (
        "中国公募赎回引致的股票卖压代理"
    )
    assert factor_display_name("alpha101_054") == "开收比五次方加权负收盘位置"
    assert factor_display_name("alpha191_046") == "3／6／12／24日均线综合价格比"
    assert factor_display_name("feature_cn_redemption_stock_pressure") == (
        "处理后·中国公募赎回引致的股票卖压代理"
    )
    assert factor_display_name("feature_pre_neutral_reversal_5d") == (
        "中性化前·5日·价格反转"
    )
