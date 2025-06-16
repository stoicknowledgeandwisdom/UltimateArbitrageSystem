terraform {
  source = "./"
}

inputs = {
  project_name       = local.project_name
  environment        = local.environment
  aws_region         = local.aws_region
  regions           = var.regions
  domain_name       = "${local.project_name}-${local.environment}.${local.aws_region}.elb.amazonaws.com"
  git_repository_url = "https://github.com/your-org/UltimateArbitrageSystem.git"
  git_branch        = local.environment == "prod" ? "main" : local.environment
  common_tags       = local.common_tags
}

