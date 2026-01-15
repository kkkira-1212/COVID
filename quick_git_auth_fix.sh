#!/bin/bash
# Quick Git Authentication Fix Script

echo "=== Git Authentication Quick Fix ==="
echo ""

# Check if already using SSH
CURRENT_URL=$(git remote get-url origin)
if [[ $CURRENT_URL == git@* ]]; then
    echo "✓ Already using SSH!"
    echo "  Remote URL: $CURRENT_URL"
    echo ""
    echo "Testing SSH connection..."
    ssh -T git@github.com 2>&1 | grep -E "(Hi|successfully|denied|failed)" || echo "SSH connection test"
    echo ""
    echo "If SSH works, just run: git push"
    exit 0
fi

echo "Current remote uses HTTPS: $CURRENT_URL"
echo ""

# Option 1: Quick fix with PAT
echo "=== Option 1: Use Personal Access Token (Quick) ==="
echo ""
echo "Steps:"
echo "1. Configure credential helper:"
echo "   git config --global credential.helper store"
echo ""
echo "2. Push (you'll be prompted for credentials):"
echo "   git push"
echo "   When prompted:"
echo "     Username: kkkira-1212"
echo "     Password: <your PAT token>"
echo ""
echo "Note: PAT will be stored in ~/.git-credentials after first use"
echo ""

# Option 2: Switch to SSH (recommended)
echo "=== Option 2: Switch to SSH (Recommended - More Secure) ==="
echo ""

if [ -f ~/.ssh/id_ed25519.pub ] || [ -f ~/.ssh/id_rsa.pub ]; then
    echo "✓ SSH key found!"
    echo ""
    echo "Your public key:"
    cat ~/.ssh/id_*.pub 2>/dev/null | head -1
    echo ""
    echo "Next steps:"
    echo "1. Add this public key to GitHub: https://github.com/settings/keys"
    echo "2. Switch to SSH: git remote set-url origin git@github.com:kkkira-1212/COVID.git"
    echo "3. Test: ssh -T git@github.com"
    echo "4. Push: git push"
else
    echo "✗ No SSH key found."
    echo ""
    echo "Would you like to generate SSH key? (Y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]] || [[ -z "$response" ]]; then
        echo "Generating SSH key..."
        ssh-keygen -t ed25519 -C "wangxin2004001@163.com" -f ~/.ssh/id_ed25519 -N ""
        echo ""
        echo "✓ SSH key generated!"
        echo ""
        echo "Your public key (copy this and add to GitHub):"
        cat ~/.ssh/id_ed25519.pub
        echo ""
        echo "Add this key to GitHub: https://github.com/settings/keys"
        echo "Then run:"
        echo "  git remote set-url origin git@github.com:kkkira-1212/COVID.git"
        echo "  ssh -T git@github.com"
        echo "  git push"
    fi
fi





