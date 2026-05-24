import {
  type DragEvent,
  useState,
} from 'react'
import { FieldLabel } from '../../shared/ui/components'

export function CsvUploadField({
  csvFileName,
  onFileChange,
}: {
  csvFileName: string | undefined
  onFileChange: (file: File | null) => void | Promise<void>
}) {
  const [dragActive, setDragActive] = useState(false)
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault()
    setDragActive(false)
    void onFileChange(event.dataTransfer.files[0] ?? null)
  }
  return (
    <label
      aria-label="拖拽上传 CSV"
      className={
        dragActive ? 'field wide file-field drag-active' : 'field wide file-field'
      }
      onDragLeave={() => setDragActive(false)}
      onDragOver={(event) => {
        event.preventDefault()
        setDragActive(true)
      }}
      onDrop={handleDrop}
    >
      <FieldLabel fieldName="factor CSV" label="上传文件" required />
      <span className="file-prompt">{csvFileName ?? '选择或拖入 CSV 文件'}</span>
      <input
        accept=".csv,text/csv"
        aria-label="上传文件 factor CSV"
        className="file-input"
        onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        type="file"
      />
    </label>
  )
}
