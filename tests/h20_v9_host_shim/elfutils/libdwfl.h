#pragma once
// Host-only shim: deep_jit's exception backtrace dlopens libdw at runtime, so
// declarations are enough for the local heuristic gate to compile.
#include <cstdint>
#include <sys/types.h>
typedef uint64_t Dwarf_Addr;
typedef uint32_t GElf_Word;
struct Dwfl; struct Dwfl_Module; struct Dwfl_Line;
typedef struct {
    int (*find_elf)(Dwfl_Module*, void**, const char*, Dwarf_Addr, char**, void**);
    int (*find_debuginfo)(Dwfl_Module*, void**, const char*, Dwarf_Addr, const char*, const char*, GElf_Word, char**);
    void* section_address; char** debuginfo_path;
} Dwfl_Callbacks;
extern "C" {
Dwfl* dwfl_begin(const Dwfl_Callbacks*);
int dwfl_linux_proc_report(Dwfl*, pid_t);
int dwfl_report_end(Dwfl*, int (*)(Dwfl_Module*, void*, const char*, Dwarf_Addr, void*), void*);
Dwfl_Module* dwfl_addrmodule(Dwfl*, Dwarf_Addr);
const char* dwfl_module_addrname(Dwfl_Module*, Dwarf_Addr);
Dwfl_Line* dwfl_module_getsrc(Dwfl_Module*, Dwarf_Addr);
const char* dwfl_lineinfo(Dwfl_Line*, Dwarf_Addr*, int*, int*, int*, const char**);
int dwfl_linux_proc_find_elf(Dwfl_Module*, void**, const char*, Dwarf_Addr, char**, void**);
}
