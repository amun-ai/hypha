{{- define "hypha.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "namespace" -}}
{{- .Values.namespace | default .Release.Namespace -}}
{{- end -}}


{{- define "hypha.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "triton.url" -}}
{{- if (index .Values "tritoninfereceserver-hypha" "enabled") -}}
{{- printf "http://tritoninferenceserver.%s.svc.cluster.local:%s" (.Values.namespace | default .Release.Namespace) "8000" -}}
{{- else -}}
{{ .Values.triton_url}}
{{- end -}}
{{- end -}}
