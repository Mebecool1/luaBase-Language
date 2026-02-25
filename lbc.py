
"""
LBC - Custom Language Compiler
Compiler that converts .lbcode source files to .lbvm bytecode
"""

import sys
import re
from enum import Enum, auto
from typing import List, Dict, Tuple, Any, Optional

# Token types
class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords
    IF = auto()
    THEN = auto()
    ELSE = auto()
    END = auto()
    WHILE = auto()
    DO = auto()
    FOR = auto()
    IN = auto()
    FUNCTION = auto()
    RETURN = auto()
    IMPORT = auto()
    FROM = auto()
    PRINT = auto()
    SCANF = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    EXTERN = auto()
    
    # Type keywords
    INT = auto()
    STR = auto()
    FLOAT = auto()
    
    # Operators
    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    DOT = auto()
    STAR = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()

class Token:
    def __init__(self, token_type: TokenType, value: Any, line: int, col: int):
        self.type = token_type
        self.value = value
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        
        self.keywords = {
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'else': TokenType.ELSE,
            'end': TokenType.END,
            'while': TokenType.WHILE,
            'do': TokenType.DO,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'function': TokenType.FUNCTION,
            'return': TokenType.RETURN,
            'import': TokenType.IMPORT,
            'from': TokenType.FROM,
            'print': TokenType.PRINT,
            'scanf': TokenType.SCANF,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'int': TokenType.INT,
            'str': TokenType.STR,
            'float': TokenType.FLOAT,
            'extern': TokenType.EXTERN,
        }
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset=1) -> Optional[str]:
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self):
        if self.pos < len(self.source):
            if self.source[self.pos] == '\n':
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '#':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_string(self, quote_char):
        start_line, start_col = self.line, self.col
        self.advance()  # Skip opening quote
        value = ""
        
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if self.current_char() == 'n':
                    value += '\n'
                elif self.current_char() == 't':
                    value += '\t'
                elif self.current_char() == 'r':
                    value += '\r'
                elif self.current_char() == '\\':
                    value += '\\'
                elif self.current_char() == '"':
                    value += '"'
                elif self.current_char() == "'":
                    value += "'"
                else:
                    value += self.current_char()
                self.advance()
            else:
                value += self.current_char()
                self.advance()
        
        if not self.current_char():
            raise SyntaxError(f"Unterminated string at {start_line}:{start_col}")
        
        self.advance()  # Skip closing quote
        return value
    
    def read_number(self):
        value = ""
        has_dot = False
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                if has_dot:
                    break
                has_dot = True
            value += self.current_char()
            self.advance()
        
        if has_dot:
            return float(value)
        else:
            return int(value)
    
    def read_identifier(self):
        value = ""
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            value += self.current_char()
            self.advance()
        return value
    
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            if self.current_char() == '#':
                self.skip_comment()
                continue
            
            line, col = self.line, self.col
            char = self.current_char()
            
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', line, col))
                self.advance()
            elif char == '"' or char == "'":
                value = self.read_string(char)
                self.tokens.append(Token(TokenType.STRING, value, line, col))
            elif char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, line, col))
            elif char.isalpha() or char == '_':
                ident = self.read_identifier()
                token_type = self.keywords.get(ident, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, ident, line, col))
            elif char == '=':
                if self.peek_char() == '=':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.EQ, '==', line, col))
                else:
                    self.advance()
                    self.tokens.append(Token(TokenType.ASSIGN, '=', line, col))
            elif char == '!':
                if self.peek_char() == '=':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.NE, '!=', line, col))
                else:
                    raise SyntaxError(f"Unexpected character '!' at {line}:{col}")
            elif char == '<':
                if self.peek_char() == '=':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.LE, '<=', line, col))
                else:
                    self.advance()
                    self.tokens.append(Token(TokenType.LT, '<', line, col))
            elif char == '>':
                if self.peek_char() == '=':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.GE, '>=', line, col))
                else:
                    self.advance()
                    self.tokens.append(Token(TokenType.GT, '>', line, col))
            elif char == '+':
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, '+', line, col))
            elif char == '-':
                self.advance()
                self.tokens.append(Token(TokenType.MINUS, '-', line, col))
            elif char == '*':
                self.advance()
                self.tokens.append(Token(TokenType.MULTIPLY, '*', line, col))
            elif char == '/':
                self.advance()
                self.tokens.append(Token(TokenType.DIVIDE, '/', line, col))
            elif char == '%':
                self.advance()
                self.tokens.append(Token(TokenType.MODULO, '%', line, col))
            elif char == '(':
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, '(', line, col))
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', line, col))
            elif char == '{':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', line, col))
            elif char == '}':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, '}', line, col))
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', line, col))
            elif char == ':':
                self.advance()
                self.tokens.append(Token(TokenType.COLON, ':', line, col))
            elif char == '.':
                self.advance()
                self.tokens.append(Token(TokenType.DOT, '.', line, col))
            else:
                raise SyntaxError(f"Unexpected character '{char}' at {line}:{col}")
        
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

# AST Node types
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements: List[ASTNode], imports: List[str] = None):
        self.statements = statements
        self.imports = imports or []

class VarDecl(ASTNode):
    def __init__(self, var_type: str, name: str, value: ASTNode = None):
        self.var_type = var_type
        self.name = name
        self.value = value

class Assignment(ASTNode):
    def __init__(self, name: str, value: ASTNode):
        self.name = name
        self.value = value

class BinaryOp(ASTNode):
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

class Number(ASTNode):
    def __init__(self, value: float):
        self.value = value

class String(ASTNode):
    def __init__(self, value: str):
        self.value = value

class Boolean(ASTNode):
    def __init__(self, value: bool):
        self.value = value

class Variable(ASTNode):
    def __init__(self, name: str):
        self.name = name

class PrintStatement(ASTNode):
    def __init__(self, args: List[ASTNode]):
        self.args = args

class ScanfStatement(ASTNode):
    def __init__(self, var_name: str):
        self.var_name = var_name

class IfStatement(ASTNode):
    def __init__(self, condition: ASTNode, then_block: List[ASTNode], else_block: List[ASTNode] = None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class WhileLoop(ASTNode):
    def __init__(self, condition: ASTNode, body: List[ASTNode]):
        self.condition = condition
        self.body = body

class ForLoop(ASTNode):
    def __init__(self, var: str, start: ASTNode, end: ASTNode, body: List[ASTNode]):
        self.var = var
        self.start = start
        self.end = end
        self.body = body

class FunctionDef(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name = name
        self.params = params
        self.body = body

class ReturnStatement(ASTNode):
    def __init__(self, value: ASTNode = None):
        self.value = value

class FunctionCall(ASTNode):
    def __init__(self, name: str, args: List[ASTNode]):
        self.name = name
        self.args = args

class ExternImport(ASTNode):
    def __init__(self, header_file: str, functions: List[str] = None, import_all: bool = False):
        self.header_file = header_file  # e.g., "raylib.h"
        self.functions = functions or []  # e.g., ["InitWindow", "DrawRectangle"]
        self.import_all = import_all  # True if using * wildcard

class StructCreation(ASTNode):
    def __init__(self, struct_name: str, args: List[ASTNode]):
        self.struct_name = struct_name  # e.g., "Vector3"
        self.args = args  # Constructor arguments

class FieldAccess(ASTNode):
    def __init__(self, object_expr: ASTNode, field_name: str):
        self.object_expr = object_expr  # The object (e.g., variable or struct instance)
        self.field_name = field_name  # Field name (e.g., "x", "y", "z")

class FieldAssignment(ASTNode):
    def __init__(self, object_expr: ASTNode, field_name: str, value: ASTNode):
        self.object_expr = object_expr
        self.field_name = field_name
        self.value = value

class Parser:
    # Known struct types from raylib.h
    KNOWN_STRUCTS = {
        "Vector2", "Vector3", "Vector4",
        "Color",
        "Rectangle",
        "Matrix",
        "Quaternion",
        "Transform",
        "Camera2D", "Camera3D",
    }
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]
    
    def peek_token(self, offset=1) -> Token:
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[pos]
    
    def advance(self):
        self.pos += 1
    
    def expect(self, token_type: TokenType) -> Token:
        if self.current_token().type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token().type} at {self.current_token().line}:{self.current_token().col}")
        token = self.current_token()
        self.advance()
        return token
    
    def skip_newlines(self):
        while self.current_token().type == TokenType.NEWLINE:
            self.advance()
    
    def parse(self) -> Program:
        statements = []
        imports = []
        extern_imports = []
        
        self.skip_newlines()
        
        # Parse imports and extern imports first
        while self.current_token().type in (TokenType.IMPORT, TokenType.EXTERN):
            if self.current_token().type == TokenType.IMPORT:
                self.advance()
                module = self.expect(TokenType.IDENTIFIER).value
                imports.append(module)
                self.skip_newlines()
            elif self.current_token().type == TokenType.EXTERN:
                extern_import = self.parse_extern_import()
                extern_imports.append(extern_import)
                self.skip_newlines()
        
        # Parse statements
        while self.current_token().type != TokenType.EOF:
            self.skip_newlines()
            if self.current_token().type == TokenType.EOF:
                break
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        # Combine extern imports as statements
        all_statements = extern_imports + statements
        program = Program(all_statements, imports)
        return program
    
    def parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        
        token_type = self.current_token().type
        
        if token_type == TokenType.IF:
            return self.parse_if_statement()
        elif token_type == TokenType.WHILE:
            return self.parse_while_loop()
        elif token_type == TokenType.FOR:
            return self.parse_for_loop()
        elif token_type == TokenType.FUNCTION:
            return self.parse_function_def()
        elif token_type == TokenType.RETURN:
            return self.parse_return_statement()
        elif token_type == TokenType.PRINT:
            return self.parse_print_statement()
        elif token_type == TokenType.SCANF:
            return self.parse_scanf_statement()
        elif token_type in (TokenType.INT, TokenType.STR, TokenType.FLOAT):
            return self.parse_var_decl()
        elif token_type == TokenType.IDENTIFIER:
            return self.parse_assignment()
        else:
            raise SyntaxError(f"Unexpected token {token_type} at {self.current_token().line}:{self.current_token().col}")
    
    def parse_extern_import(self) -> ExternImport:
        """Parse: extern from "raylib.h" import InitWindow, DrawRectangle
                  or: extern from "raylib.h" import *"""
        self.expect(TokenType.EXTERN)
        self.expect(TokenType.FROM)
        header_file = self.expect(TokenType.STRING).value
        self.expect(TokenType.IMPORT)
        
        # Check for wildcard import
        if self.current_token().type == TokenType.MULTIPLY:
            self.advance()
            return ExternImport(header_file, [], import_all=True)
        
        # Parse specific function imports
        functions = []
        while True:
            func_name = self.expect(TokenType.IDENTIFIER).value
            functions.append(func_name)
            if self.current_token().type == TokenType.COMMA:
                self.advance()
            else:
                break
        
        return ExternImport(header_file, functions, import_all=False)
    
    def parse_if_statement(self) -> IfStatement:
        self.expect(TokenType.IF)
        condition = self.parse_expression()
        self.expect(TokenType.THEN)
        self.skip_newlines()
        
        then_block = []
        while self.current_token().type not in (TokenType.ELSE, TokenType.END, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                then_block.append(stmt)
            self.skip_newlines()
        
        else_block = None
        if self.current_token().type == TokenType.ELSE:
            self.advance()
            self.skip_newlines()
            else_block = []
            while self.current_token().type not in (TokenType.END, TokenType.EOF):
                stmt = self.parse_statement()
                if stmt:
                    else_block.append(stmt)
                self.skip_newlines()
        
        self.expect(TokenType.END)
        return IfStatement(condition, then_block, else_block)
    
    def parse_while_loop(self) -> WhileLoop:
        self.expect(TokenType.WHILE)
        condition = self.parse_expression()
        self.expect(TokenType.DO)
        self.skip_newlines()
        
        body = []
        while self.current_token().type not in (TokenType.END, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        return WhileLoop(condition, body)
    
    def parse_for_loop(self) -> ForLoop:
        self.expect(TokenType.FOR)
        var_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        start = self.parse_expression()
        
        # For now, assume "to" means up to, use COMMA as separator
        if self.current_token().type == TokenType.COMMA:
            self.advance()
        else:
            raise SyntaxError(f"Expected comma in for loop at {self.current_token().line}:{self.current_token().col}")
        
        end = self.parse_expression()
        self.expect(TokenType.DO)
        self.skip_newlines()
        
        body = []
        while self.current_token().type not in (TokenType.END, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        return ForLoop(var_name, start, end, body)
    
    def parse_function_def(self) -> FunctionDef:
        self.expect(TokenType.FUNCTION)
        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)
        
        params = []
        while self.current_token().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENTIFIER).value)
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        self.skip_newlines()
        
        body = []
        while self.current_token().type not in (TokenType.END, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.expect(TokenType.END)
        return FunctionDef(name, params, body)
    
    def parse_return_statement(self) -> ReturnStatement:
        self.expect(TokenType.RETURN)
        value = None
        if self.current_token().type not in (TokenType.NEWLINE, TokenType.EOF):
            value = self.parse_expression()
        return ReturnStatement(value)
    
    def parse_print_statement(self) -> PrintStatement:
        self.expect(TokenType.PRINT)
        self.expect(TokenType.LPAREN)
        
        args = []
        while self.current_token().type != TokenType.RPAREN:
            args.append(self.parse_expression())
            if self.current_token().type == TokenType.COMMA:
                self.advance()
        
        self.expect(TokenType.RPAREN)
        return PrintStatement(args)
    
    def parse_scanf_statement(self) -> ScanfStatement:
        self.expect(TokenType.SCANF)
        self.expect(TokenType.LPAREN)
        var_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.RPAREN)
        return ScanfStatement(var_name)
    
    def parse_var_decl(self) -> VarDecl:
        var_type = self.current_token().value
        self.advance()
        name = self.expect(TokenType.IDENTIFIER).value
        
        value = None
        if self.current_token().type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expression()
        
        return VarDecl(var_type, name, value)
    
    def parse_assignment(self) -> Optional[ASTNode]:
        # Parse the left side (could be variable, function call, field access, struct creation, etc.)
        ident_name = self.expect(TokenType.IDENTIFIER).value
        
        # Check for field access
        left_expr = Variable(ident_name)
        while self.current_token().type == TokenType.DOT:
            self.advance()
            field_name = self.expect(TokenType.IDENTIFIER).value
            left_expr = FieldAccess(left_expr, field_name)
        
        # Now check what comes after
        if self.current_token().type == TokenType.LPAREN:
            # Function or struct creation call
            if isinstance(left_expr, Variable):
                self.advance()
                args = []
                while self.current_token().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    if self.current_token().type == TokenType.COMMA:
                        self.advance()
                self.expect(TokenType.RPAREN)
                
                # Check if it's a struct (check against known structs)
                if ident_name in self.KNOWN_STRUCTS:
                    return StructCreation(ident_name, args)
                else:
                    return FunctionCall(ident_name, args)
            else:
                raise SyntaxError(f"Cannot call function on field access at {self.current_token().line}:{self.current_token().col}")
        
        elif self.current_token().type == TokenType.ASSIGN:
            # Assignment
            self.advance()
            value = self.parse_expression()
            
            if isinstance(left_expr, Variable):
                return Assignment(ident_name, value)
            elif isinstance(left_expr, FieldAccess):
                return FieldAssignment(left_expr.object_expr, left_expr.field_name, value)
            else:
                raise SyntaxError(f"Invalid assignment target at {self.current_token().line}:{self.current_token().col}")
        else:
            raise SyntaxError(f"Expected '(', '=', or '.' after identifier '{ident_name}' at {self.current_token().line}:{self.current_token().col}")
    
    def parse_expression(self) -> ASTNode:
        return self.parse_or_expr()
    
    def parse_or_expr(self) -> ASTNode:
        left = self.parse_and_expr()
        
        while self.current_token().type == TokenType.OR:
            op = self.current_token().value
            self.advance()
            right = self.parse_and_expr()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_and_expr(self) -> ASTNode:
        left = self.parse_not_expr()
        
        while self.current_token().type == TokenType.AND:
            op = self.current_token().value
            self.advance()
            right = self.parse_not_expr()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_not_expr(self) -> ASTNode:
        if self.current_token().type == TokenType.NOT:
            op = self.current_token().value
            self.advance()
            operand = self.parse_not_expr()
            return UnaryOp(op, operand)
        
        return self.parse_comparison()
    
    def parse_comparison(self) -> ASTNode:
        left = self.parse_additive()
        
        while self.current_token().type in (TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.current_token().value
            self.advance()
            right = self.parse_additive()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_additive(self) -> ASTNode:
        left = self.parse_multiplicative()
        
        while self.current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token().value
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        left = self.parse_unary()
        
        while self.current_token().type in (TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            op = self.current_token().value
            self.advance()
            right = self.parse_unary()
            left = BinaryOp(left, op, right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        if self.current_token().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token().value
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> ASTNode:
        """Parse postfix operations like field access (dot notation)"""
        expr = self.parse_primary()
        
        # Handle field access and assignment
        while self.current_token().type == TokenType.DOT:
            self.advance()
            field_name = self.expect(TokenType.IDENTIFIER).value
            expr = FieldAccess(expr, field_name)
        
        return expr
    
    def parse_primary(self) -> ASTNode:
        token_type = self.current_token().type
        token_value = self.current_token().value
        
        if token_type == TokenType.NUMBER:
            self.advance()
            return Number(token_value)
        elif token_type == TokenType.STRING:
            self.advance()
            return String(token_value)
        elif token_type == TokenType.TRUE:
            self.advance()
            return Boolean(True)
        elif token_type == TokenType.FALSE:
            self.advance()
            return Boolean(False)
        elif token_type == TokenType.IDENTIFIER:
            self.advance()
            # Check for function call or struct creation
            if self.current_token().type == TokenType.LPAREN:
                self.advance()
                args = []
                while self.current_token().type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    if self.current_token().type == TokenType.COMMA:
                        self.advance()
                self.expect(TokenType.RPAREN)
                
                # Distinguish struct creation from function call
                # Check if it's a known struct type
                if token_value in self.KNOWN_STRUCTS:
                    return StructCreation(token_value, args)
                else:
                    return FunctionCall(token_value, args)
            else:
                return Variable(token_value)
        elif token_type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        else:
            raise SyntaxError(f"Unexpected token {token_type} at {self.current_token().line}:{self.current_token().col}")

# Bytecode generation
class Compiler:
    def __init__(self, ast: Program):
        self.ast = ast
        self.bytecode = []
        self.symbols = {}  # symbol table
        self.functions = {}  # function definitions
        self.extern_functions = {}  # extern C functions: {func_name: header_file}
        self.wildcard_imports = []  # List of header files with import *
        self.label_counter = 0
    
    def new_label(self) -> str:
        label = f"_L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def compile(self) -> bytes:
        # First pass: collect extern imports and function definitions
        self.extern_functions = {}
        self.functions = {}
        self.wildcard_imports = []  # List of header files with import *
        
        for stmt in self.ast.statements:
            if isinstance(stmt, ExternImport):
                if stmt.import_all:
                    self.wildcard_imports.append(stmt.header_file)
                else:
                    for func_name in stmt.functions:
                        self.extern_functions[func_name] = stmt.header_file
            elif isinstance(stmt, FunctionDef):
                self.functions[stmt.name] = stmt
        
        # Second pass: compile main statements (skip function defs and extern imports)
        for stmt in self.ast.statements:
            if not isinstance(stmt, (FunctionDef, ExternImport)):
                self.compile_statement(stmt)
        
        # Jump to halt (skip function definitions)
        halt_label = self.new_label()
        self.bytecode.append(("JMP", halt_label))
        
        # Third pass: compile function definitions
        for func_name, func_def in self.functions.items():
            self.bytecode.append(("LABEL", f"_FUNC_{func_name}"))
            # Generate parameter binding code
            # Parameters are on the stack in reverse order
            for i, param_name in enumerate(reversed(func_def.params)):
                self.bytecode.append(("STORE", param_name))
            # Compile function body
            for s in func_def.body:
                self.compile_statement(s)
            # Add implicit return only if last statement isn't already a return
            if not (func_def.body and isinstance(func_def.body[-1], ReturnStatement)):
                self.bytecode.append("RETURN")
        
        # Halt label
        self.bytecode.append(("LABEL", halt_label))
        
        # Add halt instruction
        self.bytecode.append("HALT")
        
        # Convert to bytecode format
        return self.serialize_bytecode()
    
    def serialize_bytecode(self) -> bytes:
        """Convert bytecode to binary format"""
        import json
        bytecode_str = json.dumps(self.bytecode)
        return bytecode_str.encode('utf-8')
    
    def compile_statement(self, stmt: ASTNode):
        if isinstance(stmt, VarDecl):
            self.compile_var_decl(stmt)
        elif isinstance(stmt, Assignment):
            self.compile_assignment(stmt)
        elif isinstance(stmt, FieldAssignment):
            self.compile_field_assignment(stmt)
        elif isinstance(stmt, PrintStatement):
            self.compile_print(stmt)
        elif isinstance(stmt, ScanfStatement):
            self.compile_scanf(stmt)
        elif isinstance(stmt, IfStatement):
            self.compile_if(stmt)
        elif isinstance(stmt, WhileLoop):
            self.compile_while(stmt)
        elif isinstance(stmt, ForLoop):
            self.compile_for(stmt)
        elif isinstance(stmt, FunctionDef):
            pass  # Already collected
        elif isinstance(stmt, ReturnStatement):
            self.compile_return(stmt)
        elif isinstance(stmt, FunctionCall):
            self.compile_function_call(stmt)
        elif isinstance(stmt, ExternImport):
            self.compile_extern_import(stmt)
    
    def compile_var_decl(self, stmt: VarDecl):
        self.symbols[stmt.name] = stmt.var_type
        if stmt.value:
            self.compile_expression(stmt.value)
            self.bytecode.append(("STORE", stmt.name))
        else:
            # Initialize with default value
            if stmt.var_type == "int":
                self.bytecode.append(("PUSH", 0))
            elif stmt.var_type == "float":
                self.bytecode.append(("PUSH", 0.0))
            elif stmt.var_type == "str":
                self.bytecode.append(("PUSH", ""))
            self.bytecode.append(("STORE", stmt.name))
    
    def compile_assignment(self, stmt: Assignment):
        self.compile_expression(stmt.value)
        self.bytecode.append(("STORE", stmt.name))
    
    def compile_print(self, stmt: PrintStatement):
        for arg in stmt.args:
            self.compile_expression(arg)
            self.bytecode.append("PRINT")
    
    def compile_scanf(self, stmt: ScanfStatement):
        self.bytecode.append(("INPUT", stmt.var_name))
        self.bytecode.append(("STORE", stmt.var_name))
    
    def compile_if(self, stmt: IfStatement):
        self.compile_expression(stmt.condition)
        else_label = self.new_label()
        end_label = self.new_label()
        
        self.bytecode.append(("JIF", else_label))
        
        for s in stmt.then_block:
            self.compile_statement(s)
        
        self.bytecode.append(("JMP", end_label))
        self.bytecode.append(("LABEL", else_label))
        
        if stmt.else_block:
            for s in stmt.else_block:
                self.compile_statement(s)
        
        self.bytecode.append(("LABEL", end_label))
    
    def compile_while(self, stmt: WhileLoop):
        loop_label = self.new_label()
        end_label = self.new_label()
        
        self.bytecode.append(("LABEL", loop_label))
        self.compile_expression(stmt.condition)
        self.bytecode.append(("JIF", end_label))
        
        for s in stmt.body:
            self.compile_statement(s)
        
        self.bytecode.append(("JMP", loop_label))
        self.bytecode.append(("LABEL", end_label))
    
    def compile_for(self, stmt: ForLoop):
        loop_label = self.new_label()
        end_label = self.new_label()
        
        # Initialize loop variable
        self.symbols[stmt.var] = "int"
        self.compile_expression(stmt.start)
        self.bytecode.append(("STORE", stmt.var))
        
        self.bytecode.append(("LABEL", loop_label))
        # Load var, load end value, compare (var <= end)
        self.bytecode.append(("LOAD", stmt.var))
        self.compile_expression(stmt.end)
        self.bytecode.append(("<=",))
        # Jump to end if condition is false
        self.bytecode.append(("JIF", end_label))
        
        # Loop body
        for s in stmt.body:
            self.compile_statement(s)
        
        # Increment loop variable
        self.bytecode.append(("LOAD", stmt.var))
        self.bytecode.append(("PUSH", 1))
        self.bytecode.append(("ADD",))
        self.bytecode.append(("STORE", stmt.var))
        
        # Jump back to loop check
        self.bytecode.append(("JMP", loop_label))
        self.bytecode.append(("LABEL", end_label))
    
    def compile_return(self, stmt: ReturnStatement):
        if stmt.value:
            self.compile_expression(stmt.value)
        else:
            self.bytecode.append(("PUSH", None))
        self.bytecode.append("RETURN")
    
    def compile_function_call(self, expr: FunctionCall):
        # Push arguments onto stack
        for arg in expr.args:
            self.compile_expression(arg)
        
        # Check if it's an explicit extern C function
        if expr.name in self.extern_functions:
            header_file = self.extern_functions[expr.name]
            self.bytecode.append(("EXTERN_CALL", expr.name, header_file, len(expr.args)))
        # Check if it's a function defined in this program
        elif expr.name in self.functions:
            self.bytecode.append(("CALL", expr.name, len(expr.args)))
        # Otherwise, check if there are wildcard imports
        elif self.wildcard_imports:
            # Try wildcard imports at runtime
            self.bytecode.append(("EXTERN_WILDCARD_CALL", expr.name, self.wildcard_imports[:], len(expr.args)))
        else:
            # Regular function call (will fail at runtime if not found)
            self.bytecode.append(("CALL", expr.name, len(expr.args)))
    
    def compile_extern_import(self, stmt: ExternImport):
        """Generate bytecode for extern import (handled during compilation pass)"""
        # Extern imports are collected in the first pass of compile()
        # No bytecode needed here
        pass
    
    def compile_expression(self, expr: ASTNode):
        if isinstance(expr, Number):
            self.bytecode.append(("PUSH", expr.value))
        elif isinstance(expr, String):
            self.bytecode.append(("PUSH", expr.value))
        elif isinstance(expr, Boolean):
            self.bytecode.append(("PUSH", expr.value))
        elif isinstance(expr, Variable):
            self.bytecode.append(("LOAD", expr.name))
        elif isinstance(expr, BinaryOp):
            self.compile_expression(expr.left)
            self.compile_expression(expr.right)
            self.bytecode.append((expr.op,))
        elif isinstance(expr, UnaryOp):
            self.compile_expression(expr.operand)
            self.bytecode.append((expr.op,))
        elif isinstance(expr, FunctionCall):
            self.compile_function_call(expr)
        elif isinstance(expr, StructCreation):
            self.compile_struct_creation(expr)
        elif isinstance(expr, FieldAccess):
            self.compile_field_access(expr)
    
    def compile_struct_creation(self, expr: StructCreation):
        """Compile struct instantiation"""
        # Push arguments onto stack
        for arg in expr.args:
            self.compile_expression(arg)
        # Generate struct creation instruction
        self.bytecode.append(("STRUCT_CREATE", expr.struct_name, len(expr.args)))
    
    def compile_field_access(self, expr: FieldAccess):
        """Compile field access (reading a field value)"""
        # Compile the object expression
        self.compile_expression(expr.object_expr)
        # Generate field get instruction
        self.bytecode.append(("STRUCT_GET", expr.field_name))
    
    def compile_field_assignment(self, stmt: FieldAssignment):
        """Compile field assignment"""
        # Load the object
        self.compile_expression(stmt.object_expr)
        # Compile the value to assign
        self.compile_expression(stmt.value)
        # Generate field set instruction
        self.bytecode.append(("STRUCT_SET", stmt.field_name))

def load_imports(ast: Program) -> Program:
    """Load and merge imported modules"""
    all_functions = []
    all_statements = list(ast.statements)
    
    for module_name in ast.imports:
        try:
            # Try to load module
            module_file = module_name + ".lb"
            with open(module_file, 'r') as f:
                module_code = f.read()
            
            # Compile module
            lexer = Lexer(module_code)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            module_ast = parser.parse()
            
            # Extract functions from module
            for stmt in module_ast.statements:
                if isinstance(stmt, FunctionDef):
                    all_functions.append(stmt)
        except FileNotFoundError:
            print(f"Warning: Module '{module_name}.lb' not found", file=sys.stderr)
    
    # Add imported functions to statements
    all_statements = all_functions + all_statements
    return Program(all_statements, [])

def compile_file(source_code: str, output_file: str):
    """Main compilation function"""
    try:
        # Tokenize
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Load imports
        ast = load_imports(ast)
        
        # Compile
        compiler = Compiler(ast)
        bytecode = compiler.compile()
        
        # Write bytecode to file
        with open(output_file, 'wb') as f:
            f.write(bytecode)
        
        print(f"Compiled successfully to {output_file}")
        return True
    except Exception as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: lbc.py <input_file> <output_file>")
        sys.exit(1)
    
    source_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        with open(source_file, 'r') as f:
            source = f.read()
        compile_file(source, output_file)
    except FileNotFoundError:
        print(f"Error: File '{source_file}' not found", file=sys.stderr)
        sys.exit(1)
