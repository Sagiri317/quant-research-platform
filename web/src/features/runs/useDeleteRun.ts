import { useState } from 'react'
import { type RunSummary, deleteRun } from '../../shared/api/api'
import { messageFrom } from '../../shared/format'

export function useDeleteRun({
  canDelete,
  onDeleted,
  onError,
  onStart,
  selectedRun,
}: {
  canDelete: boolean
  onDeleted: (publicRunId: string) => void
  onError: (message: string) => void
  onStart: () => void
  selectedRun: RunSummary | null
}) {
  const [deleting, setDeleting] = useState(false)

  async function handleDeleteSelectedRun() {
    if (!selectedRun || !canDelete) {
      return
    }
    const confirmed = window.confirm(
      [
        `确认删除回测 ${selectedRun.public_run_id}？`,
        '仅允许删除用户上传面板（data_stage=uploaded_panel）且位于当前运行根目录（run root）下的本地结果目录。',
      ].join('\n'),
    )
    if (!confirmed) {
      return
    }
    try {
      setDeleting(true)
      onStart()
      await deleteRun(selectedRun)
      onDeleted(selectedRun.public_run_id)
    } catch (caught) {
      onError(messageFrom(caught))
    } finally {
      setDeleting(false)
    }
  }

  return { deleting, handleDeleteSelectedRun }
}
