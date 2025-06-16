terraform {
  source = "./"
}

inputs = {
  cluster_name = "${local.project_name}-${local.environment}-${local.aws_region}"
  cluster_version = "1.28"
  
  # Node groups configuration
  node_groups = {
    main = {
      instance_types = [var.instance_type]
      scaling_config = {
        desired_size = var.desired_capacity
        max_size     = var.max_capacity
        min_size     = var.min_capacity
      }
      
      update_config = {
        max_unavailable_percentage = 25
      }
      
      labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
      
      taints = []
    }
  }
  
  # Addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Enable IRSA
  enable_irsa = true
  
  # CloudWatch logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # Security groups
  cluster_additional_security_group_ids = []
  
  # Enable cluster encryption
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ]
}

