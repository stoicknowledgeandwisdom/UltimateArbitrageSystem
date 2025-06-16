# Terragrunt root configuration
locals {
  # Common variables across all environments
  account_vars = yamldecode(file(find_in_parent_folders("account.yaml")))
  region_vars  = yamldecode(file(find_in_parent_folders("region.yaml")))
  env_vars     = yamldecode(file(find_in_parent_folders("env.yaml")))
  
  # Extract commonly used variables for easy access
  account_id    = local.account_vars.account_id
  project_name  = local.account_vars.project_name
  aws_region    = local.region_vars.aws_region
  environment   = local.env_vars.environment
  
  # Common tags
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    Region      = local.aws_region
    ManagedBy   = "terragrunt"
    Repository  = "UltimateArbitrageSystem"
  }
}

# Generate AWS provider configuration
generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14"
    }
  }
}

provider "aws" {
  region = "${local.aws_region}"
  
  default_tags {
    tags = ${jsonencode(local.common_tags)}
  }
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

provider "kubectl" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority.0.data)
  token                  = data.aws_eks_cluster_auth.cluster.token
  load_config_file       = false
}

data "aws_eks_cluster" "cluster" {
  name = "${local.project_name}-${local.environment}-${local.aws_region}"
}

data "aws_eks_cluster_auth" "cluster" {
  name = "${local.project_name}-${local.environment}-${local.aws_region}"
}
EOF
}

# Configure remote state
remote_state {
  backend = "s3"
  
  config = {
    encrypt = true
    bucket  = "${local.project_name}-terraform-state-${local.account_id}-${local.aws_region}"
    key     = "${path_relative_to_include()}/terraform.tfstate"
    region  = local.aws_region
    
    dynamodb_table = "${local.project_name}-terraform-locks"
  }
  
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite"
  }
}

# Generate common variables
generate "common_vars" {
  path      = "common_vars.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "${local.project_name}"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "${local.environment}"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "${local.aws_region}"
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = ${jsonencode(local.common_tags)}
}
EOF
}

