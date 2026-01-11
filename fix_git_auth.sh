#!/bin/bash
# Quick fix for Git authentication issue

echo "=== Git Authentication Fix ==="
echo ""
echo "Problem: GitHub no longer supports password authentication."
echo "You need to use either:"
echo "  1. Personal Access Token (PAT) - Quick fix"
echo "  2. SSH keys - More secure (recommended)"
echo ""

# Check current remote
echo "Current remote URL:"
git remote -v
echo ""

# Option 1: Use PAT with credential helper
echo "=== Option 1: Use Personal Access Token (Quick) ==="
echo ""
echo "Steps:"
echo "1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)"
echo "2. Click 'Generate new token (classic)'"
echo "3. Select scope: 'repo' (full control)"
echo "4. Copy the token"
echo ""
echo "Then run these commands:"
echo "  git config --global credential.helper store"
echo "  git push"
echo "  # When prompted:"
echo "  #   Username: kkkira-1212"
echo "  #   Password: <paste your PAT here>"
echo ""

# Option 2: Switch to SSH
echo "=== Option 2: Use SSH (Recommended) ==="
echo ""
if [ -f ~/.ssh/id_ed25519.pub ] || [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✓ SSH key found!"
    echo "Public key:"
    cat ~/.ssh/id_*.pub 2>/dev/null | head -1
    echo ""
    echo "To switch to SSH:"
    echo "  1. Add the public key above to GitHub → Settings → SSH and GPG keys"
    echo "  2. Run: git remote set-url origin git@github.com:kkkira-1212/COVID.git"
    echo "  3. Run: git push"
else
    echo "✗ No SSH key found."
    echo ""
    echo "To generate SSH key and switch:"
    echo "  1. Generate key: ssh-keygen -t ed25519 -C 'wangxin2004001@163.com'"
    echo "  2. Add to GitHub: cat ~/.ssh/id_ed25519.pub"
    echo "  3. Add key to GitHub → Settings → SSH and GPG keys"
    echo "  4. Switch remote: git remote set-url origin git@github.com:kkkira-1212/COVID.git"
    echo "  5. Test: ssh -T git@github.com"
    echo "  6. Push: git push"
fi
echo ""

