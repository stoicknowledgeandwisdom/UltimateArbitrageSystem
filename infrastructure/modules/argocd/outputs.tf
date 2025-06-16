output "argocd_namespace" {
  description = "The namespace where ArgoCD is installed"
  value       = kubernetes_namespace.argocd.metadata[0].name
}

output "argocd_server_service_name" {
  description = "The name of the ArgoCD server service"
  value       = "argocd-server"
}

output "argocd_url" {
  description = "URL to access ArgoCD UI"
  value       = "https://argocd.${var.domain_name}"
}

