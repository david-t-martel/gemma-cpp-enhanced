from gemma_cli.core.rag import RagClient

def test_rag_client():
    rag_client = RagClient("C:\\codedev\\llm\\rag-redis\\rag-binaries\\bin\\rag-server.exe")
    rag_client.start_server()
    response = rag_client.rag_command("ingest", path="C:\\codedev\\llm\\gemma_cli_project\\test.txt")
    assert response is not None
    rag_client.stop_server()
