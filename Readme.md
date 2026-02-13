## 1. 環境準備 (Prerequisites)

在執行程式碼之前，請確保您的系統已安裝 Python 3.9+。

### 設定 OpenAI API Key
請先取得 OpenAI API Key，並在執行前設定環境變數 `OPENAI_API_KEY`。

**Ubuntu / Linux (bash/zsh)**
```bash
export OPENAI_API_KEY="你的_API_Key"
```

如果希望每次開啟終端機都自動載入，請將 `export OPENAI_API_KEY="你的_API_Key"` 加到 `~/.bashrc` 或 `~/.zshrc`。

**Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="你的_API_Key"
```

**Windows (CMD)**
```cmd
set OPENAI_API_KEY=你的_API_Key
```

**Windows (永久設定，PowerShell)**
```powershell
# 使用者層級
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "你的_API_Key", "User")
# 系統層級 (需要以系統管理員身分執行)
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "你的_API_Key", "Machine")
```

**Windows (永久設定，CMD - setx)**
```cmd
:: 使用者層級
setx OPENAI_API_KEY "你的_API_Key"

:: 系統層級 (需要以系統管理員身分執行)
setx OPENAI_API_KEY "你的_API_Key" /M
```
`setx` 只會影響新開的終端機視窗，當前視窗不會立即生效。

### 安裝必要套件
請打開終端機 (Terminal / Command Prompt) 並執行以下指令：

```bash
# 基礎 LangChain 與 OpenAI 套件
pip install -U langchain langchain-openai langchain-community langchain-core langchain-classic

# 向量資料庫與資料處理
pip install faiss-cpu pandas numpy scikit-learn

# 互動式繪圖工具
pip install plotly
```
**或使用 requirements.txt 一次安裝所有套件**
```bash
pip install -r requirements.txt
```

