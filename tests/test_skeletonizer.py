from jev_proxy.config import Config
from jev_proxy.extractor import EntityFootprint
from jev_proxy.parser import Message, Turn
from jev_proxy.skeletonizer import (
    extract_go_skeleton,
    extract_python_skeleton,
    extract_ts_js_skeleton,
    generate_code_skeleton,
)
from jev_proxy.tail_optimizer import TailOptimizer


def test_python_skeletonizer():
    py_code = """
import os
import sys

class DatabaseManager:
    def __init__(self, db_url: str):
        self.url = db_url
        self.connect()

    async def execute_query(self, query: str, params: dict = None) -> list:
        # Complex multi-line database logic here
        print("Executing query...")
        return [{"id": 1}]

def health_check() -> bool:
    return True
"""
    skeleton = generate_code_skeleton(py_code, "database.py")
    assert skeleton is not None
    assert "class DatabaseManager:" in skeleton
    assert "def __init__(self, db_url: str):" in skeleton
    assert "async def execute_query(self, query: str, params: dict = None) -> list:" in skeleton
    assert "def health_check() -> bool:" in skeleton
    assert "Complex multi-line database logic" not in skeleton


def test_go_skeletonizer():
    go_code = """
package main

import "fmt"

type ServerConfig struct {
    Port int
    Host string
}

func NewServer(cfg ServerConfig) *Server {
    return &Server{cfg: cfg}
}

func (s *Server) Start() error {
    fmt.Println("Listening...")
    return nil
}
"""
    skeleton = generate_code_skeleton(go_code, "server.go")
    assert skeleton is not None
    assert "type ServerConfig struct {" in skeleton
    assert "Port int" in skeleton
    assert "func NewServer(cfg ServerConfig) *Server" in skeleton
    assert "func (s *Server) Start() error" in skeleton
    assert "fmt.Println" not in skeleton


def test_ts_skeletonizer():
    ts_code = """
export interface UserProfile {
    id: string;
    email: string;
}

export async function fetchUser(id: string): Promise<UserProfile> {
    const res = await fetch(`/api/user/${id}`);
    return res.json();
}
"""
    skeleton = generate_code_skeleton(ts_code, "user.ts")
    assert skeleton is not None
    assert "export interface UserProfile" in skeleton
    assert "export async function fetchUser" in skeleton


def test_tail_optimizer_with_skeleton():
    cfg = Config()
    opt = TailOptimizer(cfg)

    py_large_code = """
class AuthService:
    def login(self, username: str, password: str) -> str:
""" + ("        data = 'secret'\n" * 100) + """
    def verify(self, token: str) -> bool:
        return True
"""

    turns = [
        Turn(
            id=0,
            role="assistant",
            messages=[
                Message(role="assistant", content="Reading auth.py"),
                Message(role="tool", content=py_large_code),
            ],
            entities=EntityFootprint(files=["auth.py"]),
            token_count=1200,
        ),
        Turn(
            id=1,
            role="assistant",
            messages=[Message(role="assistant", content="Updated auth.py")],
            entities=EntityFootprint(files=["auth.py"]),
            token_count=50,
        ),
        Turn(
            id=2,
            role="user",
            messages=[Message(role="user", content="Now test authentication")],
            entities=EntityFootprint(),
            token_count=10,
        ),
    ]

    optimized = opt.optimize_tail(turns, "Now test authentication")
    assert len(optimized) == 3
    tool_content = optimized[0].messages[1].content_string()

    # Must contain structural outline instead of full implementation
    assert "Structural Skeleton of auth.py" in tool_content
    assert "class AuthService:" in tool_content
    assert "def login" in tool_content
    assert "def verify" in tool_content
    # Large bloated repeated lines must be gone
    assert "data = 'secret'" not in tool_content

