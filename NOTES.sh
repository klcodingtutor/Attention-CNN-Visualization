#!/bin/bash

# Step 1: Generate an SSH key pair (change the email to your GitHub email)
# -t: specifies the type of key to create (rsa is a common type)
# -b: sets the key length (4096 is recommended for security)
# -f: specifies the file path to save the key
# -C: adds a comment, usually your GitHub email
echo "Generating an SSH key pair..."
ssh-keygen -t rsa -b 4096 -C "your-email@example.com" -f ~/.ssh/github_key -N ""

# Step 2: Add the new SSH key to the SSH agent
# Start the ssh-agent in the background
echo "Starting the ssh-agent..."
eval "$(ssh-agent -s)"

# Add the private key to the ssh-agent
ssh-add ~/.ssh/github_key

# Step 3: Display the public key for you to add it to your GitHub account
# The public key is the .pub file generated earlier
echo "Copy the following SSH key and add it to your GitHub account:"
cat ~/.ssh/github_key.pub

# Step 4: Authenticate with GitHub CLI (if not already logged in)
unset GITHUB_TOKEN
echo "Authenticating with GitHub CLI..."
gh auth login
# Follow the prompts to authenticate (manual steps may be required)

# Check if the current directory is a Git repository
if [ -d .git ]; then
  echo "Inside a Git repository. Proceeding to fork..."

  # Get the repository's owner and name from the remote URL
  repo_url=$(git config --get remote.origin.url)
  
  # Extract the owner and repository name using pattern matching
  repo_name=$(basename -s .git "$repo_url")
  repo_owner=$(basename "$(dirname "$repo_url")")

  # Use GitHub CLI to fork the repository
  echo "Forking the repository: $repo_owner/$repo_name"
  gh repo fork "$repo_owner/$repo_name" --clone=false

else
  echo "This is not a Git repository. Please navigate to a valid Git repo first."
fi


# Step 7: Configure `git` to use the specific SSH key
# Create or modify the SSH config file
echo "Configuring Git to use the specific SSH key..."
cat <<EOL >> ~/.ssh/config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github_key
  IdentitiesOnly yes
EOL

# Step 8: Verify remote URLs
echo "Verifying remote URLs..."
cd repo-name
git remote -v

# Step 9: Make a change, commit, and push to your forked repository
# Example: Creating a placeholder README file
echo "Adding a README file as an example..."
echo "# My Forked Repository" > README.md
git add README.md
git commit -m "Added a README file"
git push origin main

# Done
echo "All steps completed! Your repository is now set up."
