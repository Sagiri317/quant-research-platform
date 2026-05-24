import { type FormEvent, useState } from 'react'
import { type RunSummary, createRun } from '../../shared/api/api'
import { messageFrom } from '../../shared/format'
import {
  INITIAL_CREATE_FORM,
  type CreateFormState,
  type CsvPreview,
  buildCreatePayload,
  inferCreateFormFromColumns,
  parseCsvPreview,
  validateCreateForm,
} from './runForm'

export function useCreateRun({
  onCreated,
}: {
  onCreated: (run: RunSummary) => void
}) {
  const [createForm, setCreateForm] =
    useState<CreateFormState>(INITIAL_CREATE_FORM)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [csvText, setCsvText] = useState('')
  const [csvPreview, setCsvPreview] = useState<CsvPreview | null>(null)
  const [createError, setCreateError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleCreateSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    let nextCsvText = csvText
    let nextCsvPreview = csvPreview
    if (csvFile && !nextCsvPreview) {
      nextCsvText = await csvFile.text()
      nextCsvPreview = parseCsvPreview(nextCsvText)
      setCsvText(nextCsvText)
      setCsvPreview(nextCsvPreview)
    }
    const validationError = validateCreateForm(createForm, nextCsvPreview)
    if (validationError) {
      setCreateError(validationError)
      return
    }
    try {
      setSubmitting(true)
      setCreateError('')
      const created = await createRun(buildCreatePayload(createForm, nextCsvText))
      onCreated(created)
    } catch (caught) {
      setCreateError(messageFrom(caught))
    } finally {
      setSubmitting(false)
    }
  }

  async function handleCsvFileChange(file: File | null) {
    setCsvFile(file)
    setCsvText('')
    setCsvPreview(null)
    setCreateError('')
    if (!file) {
      return
    }
    try {
      const text = await file.text()
      const preview = parseCsvPreview(text)
      setCsvText(text)
      setCsvPreview(preview)
      setCreateForm((current) => inferCreateFormFromColumns(current, preview.columns))
    } catch {
      setCreateError('CSV 预览失败，请检查文件内容。')
    }
  }

  return {
    createError,
    createForm,
    csvFileName: csvFile?.name,
    csvPreview,
    handleCreateSubmit,
    handleCsvFileChange,
    setCreateForm,
    setCreateError,
    submitting,
  }
}
