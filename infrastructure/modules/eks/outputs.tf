output "cluster_id" {
  description = "The name/id of the EKS cluster"
  value       = module.eks.cluster_id
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster OIDC Issuer"
  value       = try(module.eks.cluster_oidc_issuer_url, "")
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.eks.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.eks.cluster_platform_version
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if `enable_irsa = true`"
  value       = try(module.eks.oidc_provider_arn, "")
}

output "node_security_group_id" {
  description = "ID of the node shared security group"
  value       = module.eks.node_security_group_id
}

output "load_balancer_controller_role_arn" {
  description = "ARN of the AWS Load Balancer Controller IAM role"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

output "ebs_csi_role_arn" {
  description = "ARN of the EBS CSI IAM role"
  value       = module.ebs_csi_irsa_role.iam_role_arn
}

