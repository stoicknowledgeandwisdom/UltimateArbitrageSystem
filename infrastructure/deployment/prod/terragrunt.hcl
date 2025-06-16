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
  environment = "prod"
  instance_type = "m5.xlarge"
  min_capacity = 5
  max_capacity = 50
  desired_capacity = 10
  
  # Database configuration
  db_instance_class = "db.r5.large"
  db_allocated_storage = 500
  db_backup_retention_period = 30
  db_multi_az = true
  db_encryption = true
  
  # Feature flags
  enable_monitoring = true
  enable_canary = true
  enable_blue_green = true
  enable_security_scanning = true
  enable_compliance = true
  enable_backup = true
  
  # Multi-region setup (active-active)
  regions = ["us-west-2", "eu-west-1", "ap-southeast-1"]
  active_active = true
  
  # Error budget configuration
  error_budget_policy = "strict"
  sla_target = 99.9
  
  # OpenFeature configuration
  feature_flag_provider = "flagd"
  feature_flag_endpoint = "flagd-service.feature-flags.svc.cluster.local:8013"
}

