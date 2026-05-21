import { FACTOR_CSV_TEMPLATE_URL } from './api'

export function CreateRunGuide() {
  return (
    <section className="data-section">
      <div className="section-title">
        <h3>上传因子面板</h3>
        <a className="template-link" href={FACTOR_CSV_TEMPLATE_URL}>
          下载模板 CSV
        </a>
      </div>
      <div className="guide">
        <article className="guide-card">
          <h4>必填列</h4>
          <p>长表 CSV 至少需要</p>
          <div className="guide-tags">
            <code>timestamp</code>
            <code>asset</code>
            <code>factor</code>
            <code>forward_return</code>
            <code>forward_return_1d</code>
          </div>
        </article>
        <article className="guide-card">
          <h4>可选列</h4>
          <p>可以加入风险、分段或多个特征列；风险列只在波动率缩放权重下必填。</p>
          <div className="guide-tags">
            <code>risk</code>
            <code>split</code>
            <code>segment</code>
          </div>
        </article>
        <article className="guide-card">
          <h4>数据约束</h4>
          <p>
            timestamp 需可解析为日期时间；asset 按字符串处理；feature 和
            label 必须是数值；PnL 收益列用于生成真实回测曲线。本地平台只评估上传的 CSV。
          </p>
        </article>
      </div>
    </section>
  )
}
