# **GitHub tutorial!**



##### 1\. Clone repository (aka, download repository to your device)



**a.** Make sure you have accepted the invitation to the GitHub repository. The invitation should be in your email, or somewhere in your GitHub profile page.

**b.** Open your Terminal on your Desktop.

**c.** Run this code (you can choose your own destination folder, but this code puts the repo in your desktop)



cd Desktop

git clone https://github.com/superhector77/ManaTM

cd ManaTM



**d.** If you see the "ManaTM" folder appear in your desktop, you're good!



##### 2\. Make your first commit



**a.** Git is a collaborative coding tool. It lets many people make changes to the same code, and it is important to organize our contributions.

**b.** Run:



git checkout -b branch-name # change branch-name for a more descriptive name



**c.** Make any changes you want to the files in the folder. You can add files, remove files, and make changes to files.

**d.** Run



git status

git add .			# This adds your changes to your next commit.

git commit -m "Commit MSG"	# Replace "text" with descriptive text.

git push origin branch-name	# Replace Branch name, pushes to Github





##### 3\. Merge to Main (DANGER ZONE)



1. Go to the ManaTM page on GitHub. You’ll see a yellow bar saying "Compare \& pull request." Click it.

2\. Click "Create pull request."

3\. Once the team agrees it's good, click "Merge pull request."

4\. To get the latest updates from the team back onto your computer, run:



git checkout main

git pull origin main

