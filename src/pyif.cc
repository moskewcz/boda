#include<Python.h>
#include"boda_tu_base.H"
#include"pyif.H"

namespace boda 
{
  using namespace std;

  // simple smart pointer for PyObject *
  struct ppyo
  {
    PyObject * p;
    ppyo( void ) : p(0) { }
    ~ppyo( void ) { Py_XDECREF(p); }
    ppyo & operator = ( ppyo const & r )
    {
      Py_XDECREF( p );
      Py_XINCREF( r.p );
      p = r.p;
      return *this;
    }
    ppyo( ppyo const & r )
    {
      p = r.p;
      Py_XINCREF( r.p );
    }
    // assume null is an error, and that a generic error message is diagnostic enough (with the stack traces)
    ppyo( PyObject * const p_, bool const is_borrowed=0 ) : p(p_) 
    {
      if( !p ) { rt_py_err( "null PyObject" ); }
      if( is_borrowed ) { Py_XINCREF( p ); }
    }
    PyObject * get( void ) const { return p; }
  };

  void py_path_setup( void )
  {
    ppyo main_mod( PyImport_AddModule( "__main__" ), 1 );
    ppyo main_dict( PyModule_GetDict( main_mod.get() ), 1 );
    ppyo ret( PyRun_String("import sys,os.path;"
			   " sys.path.append(os.path.join(os.path.split("
			   "os.readlink('/proc/self/exe'))[0],'..','pysrc'))",
			   Py_file_input, main_dict.get(), main_dict.get() ) );
  }

  ppyo import_module( string const & module_name )
  {
    ppyo pName( PyString_FromString(module_name.c_str()) );
    assert_st( pName.get() );
    ppyo pModule( PyImport_Import(pName.get()) );
    return pModule;
  }

  ppyo get_callable( ppyo obj, string const & attr )
  {
    ppyo pFunc( PyObject_GetAttrString(obj.get(), attr.c_str()) );
    if( !PyCallable_Check(pFunc.get()) ) { rt_err( "attr '"+attr+"' not callable" ); }
    return pFunc;
  }

  ppyo call_obj( ppyo obj, ppyo args )
  {
    ppyo ret( PyObject_CallObject(obj.get(), args.get()) );
    return ret;
  }

  void prc_plot( void )
  {
    ppyo module = import_module( "bplot" );
    ppyo func( get_callable( module, "plot_stuff" ) );
    ppyo args( PyTuple_New(0) );
    ppyo ret = call_obj( func, args );
    printf("Result of call: %ld\n", PyInt_AsLong(ret.get()));
  }

}
