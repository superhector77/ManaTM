# **GitHub tutorial!**


<ol>

  
<li> Clone repository (aka, download repository to your device)


<ol>
<li> Make sure you have accepted the invitation to the GitHub repository. The invitation should be in your email, or somewhere in your GitHub profile page.

<li> Open your Terminal on your Desktop.

<li> Run this code (you can choose your own destination folder, but this code puts the repo in your desktop)


```
cd Desktop
git clone https://github.com/superhector77/ManaTM
cd ManaTM
```

<li> If you see the "ManaTM" folder appear in your desktop, you're good!
</ol>



<li> Make your first commit!


<ol>
<li> Git is a collaborative coding tool. It lets many people make changes to the same code, and it is important to organize our contributions.

<li> Run:


```
git checkout -b branch-name # change branch-name for a more descriptive name
```


<li> Make any changes you want to the files in the folder. You can add files, remove files, and make changes to files.

<li> Run


```
git status
git add .			# This adds your changes to your next commit.
git commit -m "Commit MSG"	# Replace "text" with descriptive text.
git push origin branch-name	# Replace Branch name, pushes to Github
```
</ol>



<li> Merge to Main (DANGER ZONE)


<ol>
<li> Go to the ManaTM page on GitHub. You’ll see a yellow bar saying "Compare \& pull request." Click it.

<li> Click "Create pull request."

<li> Once the team agrees it's good, click "Merge pull request."

<li> To get the latest updates from the team back onto your computer, run:


```
git checkout main
git pull origin main
```
</ol>
</ol>
