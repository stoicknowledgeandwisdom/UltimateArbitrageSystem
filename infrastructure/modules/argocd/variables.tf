variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "regions" {
  description = "List of regions for multi-region deployment"
  type        = list(string)
  default     = []
}

variable "domain_name" {
  description = "Domain name for ArgoCD UI"
  type        = string
  default     = "example.com"
}

variable "git_repository_url" {
  description = "Git repository URL for ArgoCD to sync from"
  type        = string
}

variable "git_branch" {
  description = "Git branch to sync from"
  type        = string
  default     = "main"
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

