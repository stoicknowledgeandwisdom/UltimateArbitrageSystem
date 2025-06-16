# Include root terragrunt configuration
include "root" {
  path = find_in_parent_folders()
}

# Include EKS module
include "eks" {
  path = "../../modules/eks/terragrunt.hcl"
}

# Include RDS module
include "rds" {
  path = "../../modules/rds/terragrunt.hcl"
}

# Include monitoring module
include "monitoring" {
  path = "../../modules/monitoring/terragrunt.hcl"
}

# Include ArgoCD module
include "argocd" {
  path = "../../modules/argocd/terragrunt.hcl"
}

# Environment-specific inputs
inputs = {
  environment = "dev"
  instance_type = "t3.medium"
  min_capacity = 2
  max_capacity = 10
  desired_capacity = 3
  
  # Database configuration
  db_instance_class = "db.t3.micro"
  db_allocated_storage = 20
  db_backup_retention_period = 7
  
  # Feature flags
  enable_monitoring = true
  enable_canary = true
  enable_blue_green = true
  
  # Multi-region setup
  regions = ["us-west-2", "eu-west-1", "ap-southeast-1"]
  
  # OpenFeature configuration
  feature_flag_provider = "flagd"
  feature_flag_endpoint = "flagd-service.feature-flags.svc.cluster.local:8013"
}

