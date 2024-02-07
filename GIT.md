# Tutorial
## Making changes
- Add/delete/change as many files as you want in the repository
- Tell Git about these changes by `add`ding them with `git add`
- Think about what group of changes you want to bundle into a `commit`
  Usually you want to *make many small commits*, such that you have as many points in time as possible to go back to.
  There is no right/wrong here, but if you have 1000 changes in a single commit, it's a bit on the large side ;)
  I like to make a commit either when
	- I'm working on a specific **task** that is now done: `Task completed`
	- I want to **back up** my progress for the given day
- Add the changes with `git commit` and continue working
  You can make as many commits as you want before `push`ing and `pull`ing
  
## Pushing and pulling
To make the others aware of your changes (and you aware of theirs), you can use `git push` and `git pull`.
The rule of thumb here is:
- **Always `git pull` before making any changes**.
  If the others have made many changes in the same files as you, it's always good to see whether you have to update something on your end by hand
- Make sure your repo is fine (`git status` will tell you so)
- Start adding your commits and do a `git push` to tell Github what changes you've made

Instead of `pull`ing, you can also just `fetch` the latest information from Github.
This will only tell you what has changed, without `merg`ing these changes into your repository.
Usually you can just `git pull` but if you've made a lot of changes already and you don't want to add anything from the others, `fetch`ing first might be a good idea.

### Merge conflicts
It can happen that there are merge conflicts from `git pull` because you're changing a file that is also changed by the others.
In this case you're stuck with this contact Eertze, hopefully he can sort it out :)

## Branches
To divide the work nicely and have different version of the code for each "major change" (think adding new classifiers, restructuring the project etc.), you can use different `branch`es.
You always want to `branch` off from the `main` branch into your own `branch` to work on.
To switch to a branch, you can simply do a `checkout` of it.

Everything above is the same regardless of the branch, the only thing is that now when you are `git push` and `git push`ing, it might not be from the same branch as the one the others are working on.
This is nice as it prevents merge conflicts and allows you to have "your own history" for the given branch.
Another thing to note is that Github will not know about branches you create locally, so you need to *set the upstream branch* (see below).

### Pull request
At some point, you are done with your major change and want to merge it back into `main` (so the others can safely start working with your changes without their code crashing).
To do so, go to Github and click on **Pull request > New pull request**.
It works like this:
- *The `base` branch should ALWAYS be `main`*, as it is the branch that you want to end up with (merge into)
- *The `compare` branch should be your own branch*, as it is the branch that you want to remove (merge from)

If it asks you to have someone review the pull request, you can put it on Eertze's name :)

# Commands
Below is an overview of a couple of the most useful Git commands and what to use them for:

| **Command** |  **Purpose** |
| --- | --- |
| `git status` | Verify what the current status of the repository is |
| `git add` | Add the changes from the given `path` to a new commit |
| `git commit` | Bundle all current changes into a commit |
| `git push` | Tell Github what changes you have made |
| `git fetch` | Get the latest information from Github |
| `git merge` | Combine your version with that of Github |
| `git pull` | Combination of `git fetch` and `git merge` |
| `git branch` | Create/list/delete branches |
| `git checkout` | Transfer to a branch |

## Examples

| **Command** |  **Purpose** |
| --- | --- |
| `git status` | **Single most useful command**. <br> It can probably explain the current state of the code much better than I can :) |
| *Adding* | |
| `git add my_file.txt` |Add the changes just the `my_file.txt` file |
| `git add .` | Add the changes from all files in the current folder (recursive) |
| `git add *` | Add the changes from all files in the current folder (not recursive) |
| *Commiting* | |
| `git commit` | Add all current changes to the commit (will open editor to put message into) |
| `git commit -m "<message>"` | Add all current changes to the commit with the `message` message |
| *Pulling* | |
| `git fetch` | Get the latest changes from Github | 
| `git merge` | Integrate the Github changes into your repository | 
| `git pull` | Combination of getting and integrating the latest changes from Github | 
| *Pushing* | |
| `git push` | Push all commits from your branch to the corresponding branch on Github |
| `git push --set-upstream origin <branch>` | Push all commits from your branch to a new `branch` on Github. <br> Keeping the same name as your local branch is probably the best |
| *Branching* | |
| `git branch` | List all active branches |
| `git branch -d <branch>` | Delete local `branch` (you need to be on another branch to do this) |
| `git branch --unset-upstream <branch>` | Delete Github `branch` (**This deletes it for everyone, so be careful!**) |
| `git checkout <branch>` | Go to existing `branch` |
| `git checkout -b <branch>` | Create `branch` and go to it |