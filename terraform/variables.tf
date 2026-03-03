# ─────────────────────────────────────────────
# variables.tf — Terraform Variables
# ─────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name — used for tagging all resources"
  type        = string
  default     = "spaceship-titanic-app"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "key_pair_name" {
  description = "Name of your existing AWS EC2 key pair (without .pem)"
  type        = string
  # No default — must be provided in terraform.tfvars
}

variable "github_repo_url" {
  description = "GitHub repo URL to clone on EC2"
  type        = string
  # No default — must be provided in terraform.tfvars
}

variable "repo_branch" {
  description = "Git branch to checkout"
  type        = string
  default     = "main"
}
