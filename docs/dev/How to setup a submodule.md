Sure! Here is the translation of the answer:

To add a submodule to an existing Git repository after the fact, follow these steps:

1. **Open your project** in PyCharm or your preferred editor.

2. **Open the terminal console** in PyCharm:
   - Go to `View` > `Tool Windows` > `Terminal`.

3. **Add the submodule**:
   Use the following command to add the submodule. Replace `<repository-url>` and `<path/to/submodule>` accordingly:

   ```bash
   git submodule add <repository-url> <path/to/submodule>
   ```

   Example:
   ```bash
   git submodule add https://github.com/example/repo.git submodules/repo
   ```

4. **Commit the changes**:
   After adding the submodule, you need to commit the changes:

   ```bash
   git commit -m "Add submodule"
   ```

5. **Initialize and update the submodules** (optional):
   If you want to use the submodule immediately, run these commands:

   ```bash
   git submodule init
   git submodule update
   ```

With these steps, you have successfully added the submodule to your repository after the fact.