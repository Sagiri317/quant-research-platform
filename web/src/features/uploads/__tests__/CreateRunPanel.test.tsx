import '@testing-library/jest-dom/vitest'
import { render, screen } from '@testing-library/react'
import { describe, expect, test, vi } from 'vitest'
import { CreateRunPanel } from '../CreateRunPanel'
import { INITIAL_CREATE_FORM } from '../runForm'

describe('CreateRunPanel', () => {
  test('labels CSV upload as a legacy developer smoke path', () => {
    render(
      <CreateRunPanel
        createError=""
        csvFileName={undefined}
        csvPreview={{
          columns: [
            'timestamp',
            'asset',
            'factor',
            'forward_return',
            'forward_return_1d',
          ],
          rowCount: 1,
        }}
        form={INITIAL_CREATE_FORM}
        onFileChange={vi.fn()}
        onFormChange={vi.fn()}
        onSubmit={vi.fn()}
        submitting={false}
      />,
    )

    expect(screen.getByText('开发用 CSV 上传')).toBeInTheDocument()
    expect(screen.getByText(/uploaded-panel 兼容样例/)).toBeInTheDocument()
    expect(screen.getByText(/不代表正式 A-share 私有数据运行链路/)).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: '生成开发样例' }),
    ).toBeInTheDocument()
    expect(screen.queryByText('生成回测')).not.toBeInTheDocument()
  })
})
