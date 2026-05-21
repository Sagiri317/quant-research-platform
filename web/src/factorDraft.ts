export type FactorDraft = {
  dependencies: string[]
  expression: string
  templateId: string
  templateName: string
}

export const DEFAULT_FACTOR_DRAFT: FactorDraft = {
  dependencies: ['close'],
  expression: 'neg(return(close, 20))',
  templateId: 'reversal_20d',
  templateName: '20 日反转',
}
