#оформление
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import *
from tkinter import ttk
import sys
from borders import mu1_main, mu2_main, mu3_main, mu4_main,\
                                f_main, f_test, mu1_test, mu2_test, mu3_test, mu4_test  
from grid_calculation import start_main, start_test

def table(tab,tab2_2,tab3_3,f,mu1,mu2,mu3,mu4):
    if tab==tab1:
        u,u_control,u_u,x_3d,y_3d,counter_iterations, counter_iterations_half, eps_max, norm_r, norm_z,R0=start_main(int(txt_n.get()),int(txt_m.get()),int(txt_N_max.get()),float(txt_eps.get()),
                        float(txt_a.get()),float(txt_b.get()),float(txt_c.get()),float(txt_d.get()), 
                        Var_1.get(),f_main,mu1_main,mu2_main,mu3_main,mu4_main, int(txt_k.get()))
        
    if tab==tab2:
        u,u_control,u_u,x_3d,y_3d,counter_iterations, eps_max, norm_r, norm_z,R0=start_test(int(txt_n_2.get()),int(txt_m_2.get()),int(txt_N_max_2.get()),float(txt_eps_2.get()),
                        float(txt_a_2.get()),float(txt_b_2.get()),float(txt_c_2.get()),float(txt_d_2.get()), 
                        Var_2.get(),f_test,mu1_test,mu2_test,mu3_test,mu4_test, int(txt_k_2.get()))
    table_u=[[i] for i in range (len(u))]
    for i in range (len(u)):
        for j in range(len(u[i])):
            table_u[i].append(u[i][j])

    table2_u=[[i] for i in range (len(u_control))]
    for i in range (len(u_control)):
        for j in range(len(u_control[i])):
            table2_u[i].append(u_control[i][j])
    
    table3_u=[[i] for i in range (len(u_u))]
    for i in range (len(u_u)):
        for j in range(len(u_u[i])):
            table3_u[i].append(u_u[i][j])

    heads=[]
    for i in range(len(table_u[i])):
        heads.append(str(i))

    ## основное окно    
    table = ttk.Treeview(tab, show='headings', columns=heads, height=13)
    table.place(x='25', y='5', relwidth=0.95)

    table.heading("0", text="V (i\j)", anchor='w')
    table.column('#1', width=50)
    for i in range(1, len(table_u[0])):
        j=str(i-1)
        table.heading(str(i), text=j, anchor='w')
        table.column('#'+str(i), width=100)
    
    for person in table_u:
        table.insert("", END, values=person)
   
    scrollbary = Scrollbar(tab, orient=VERTICAL)
    scrollbarx = Scrollbar(tab, orient=HORIZONTAL)
    table.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbarx.configure(command=table.xview)
    scrollbary.configure(command=table.yview)
    scrollbary.place(x=5, y=30, width=15, height=250)  
    scrollbarx.place(x=650, y= 300, width=250, height=15)

    fig=plt.figure(figsize=(6,4))   
    ax=fig.add_subplot(projection='3d')
    xgrid,ygrid=np.meshgrid(x_3d,y_3d)
    u=np.matrix(u)
    ax.plot_surface(xgrid,ygrid,u)
    
    chart=FigureCanvasTkAgg(fig,tab)
    chart.get_tk_widget().place(x='850', y='320')
    ax.set_xlabel('x', color='gray')
    ax.set_ylabel('y',color='gray')
    ax.set_zlabel('z',color='gray')

    ###второе окно
    table = ttk.Treeview(tab2_2, show='headings', columns=heads, height=13)
    table.place(x='25', y='5', relwidth=0.95)

    table.heading("0", text="V (i\j)", anchor='w')
    table.column('#1', width=50)
    for i in range(1, len(table2_u[0])):
        j=str(i-1)
        table.heading(str(i), text=j, anchor='w')
        table.column('#'+str(i), width=100)
    
    for person in table2_u:
        table.insert("", END, values=person)
   
    scrollbary = Scrollbar(tab2_2, orient=VERTICAL)
    scrollbarx = Scrollbar(tab2_2, orient=HORIZONTAL)
    table.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbarx.configure(command=table.xview)
    scrollbary.configure(command=table.yview)
    scrollbary.place(x=5, y=30, width=15, height=250)  
    scrollbarx.place(x=650, y= 300, width=250, height=15)

    fig=plt.figure(figsize=(6,4))   
    ax=fig.add_subplot(projection='3d')
    xgrid,ygrid=np.meshgrid(x_3d,y_3d)
    u_control=np.matrix(u_control)
    ax.plot_surface(xgrid,ygrid,u_control)
    
    chart=FigureCanvasTkAgg(fig,tab2_2)
    chart.get_tk_widget().place(x='850', y='320')
    ax.set_xlabel('x', color='gray')
    ax.set_ylabel('y',color='gray')
    ax.set_zlabel('z',color='gray')

    ###третье окно 
    table = ttk.Treeview(tab3_3, show='headings', columns=heads, height=13)
    table.place(x='25', y='5', relwidth=0.95)

    table.heading("0", text="V (i\j)", anchor='w')
    table.column('#1', width=50)
    for i in range(1, len(table3_u[0])):
        j=str(i-1)
        table.heading(str(i), text=j, anchor='w')
        table.column('#'+str(i), width=100)
    
    for person in table3_u:
        table.insert("", END, values=person)
   
    scrollbary = Scrollbar(tab3_3, orient=VERTICAL)
    scrollbarx = Scrollbar(tab3_3,orient=HORIZONTAL)
    table.configure(yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbarx.configure(command=table.xview)
    scrollbary.configure(command=table.yview)
    scrollbary.place(x=5, y=30, width=15, height=250)  
    scrollbarx.place(x=650, y= 300, width=250, height=15)

    fig=plt.figure(figsize=(6,4))   
    ax=fig.add_subplot(projection='3d')
    xgrid,ygrid=np.meshgrid(x_3d,y_3d)
    u_u=np.matrix(u_u)
    ax.plot_surface(xgrid,ygrid,u_u)
    
    chart=FigureCanvasTkAgg(fig,tab3_3)
    chart.get_tk_widget().place(x='850', y='320')
    ax.set_xlabel('x', color='gray')
    ax.set_ylabel('y',color='gray')
    ax.set_zlabel('z',color='gray')

    win=Tk()
    win.title('Информация')
    win.geometry('500x500')
    win_tab=ttk.Notebook(win)
    win_txt=Label(win_tab, text=("Количество итераций: "+str(counter_iterations)))
    win_txt.place(x='5',y='15')
    win_txt=Label(win_tab, text="Итоговая точность: "+str(eps_max))
    win_txt.place(x='5',y='45')
    win_txt=Label(win_tab, text="Схема на сетке решена с невязкой: "+str(norm_r))
    win_txt.place(x='5',y='75')
    win_txt=Label(win_tab, text="Невязка слау на начальном приближении: "+str(R0))
    win_txt.place(x='5',y='105')
    if tab==tab1:
        
        win_txt=Label(win_tab, text="Количество итераций при половинном шаге: "+str(counter_iterations_half,))
        win_txt.place(x='5',y='135')

    win_tab.pack(expand=1, fill='both') 
    win.mainloop()

    

def exit():
    sys. exit()  

window = Tk()
window.title("Линев, лабораторная работа по численным методам")
window.geometry('1500x800')

tab_control = ttk.Notebook(window)  
tab1_osn = ttk.Notebook(tab_control)  
tab2_osn = ttk.Notebook(tab_control)  
tab_control.add(tab1_osn, text = 'Основная задача')  
tab_control.add(tab2_osn, text = 'Тестовая задача')  

tab1=ttk.Frame(tab1_osn) 
tab1_2=ttk.Frame(tab1_osn) 
tab1_3=ttk.Frame(tab1_osn) 
tab1_osn.add(tab1, text='Полный шаг')
tab1_osn.add(tab1_2, text='Половинный шаг')
tab1_osn.add(tab1_3, text='Разница')

tab2=ttk.Frame(tab2_osn) 
tab2_2=ttk.Frame(tab2_osn) 
tab2_3=ttk.Frame(tab2_osn) 
tab2_osn.add(tab2, text='Численное решение')
tab2_osn.add(tab2_2, text='Точное решение')
tab2_osn.add(tab2_3, text='Разница')

my_canv_1 = Canvas(tab1, width = 1500, height = 300, bg = 'white'). grid(row = '0', column = '0', columnspan = '50')
my_canv_2 = Canvas(tab2, width = 1500, height = 300, bg = 'white'). grid(row = '0', column = '0', columnspan = '50')

btn_osn = Button(tab1, text="Нажмите на кнопку, чтобы построить решение", 
                 command=lambda: table(tab1,tab1_2, tab1_3,f_main, mu1_main, mu2_main,mu3_main,mu4_main,))
btn_osn.grid(row = '1', column = '0')

btn_test = Button(tab2, text="Нажмите на кнопку, чтобы построить решение", 
                  command=lambda: table(tab2,tab2_2, tab2_3,f_test,mu1_test,mu2_test,mu3_test,mu4_test))
btn_test.grid(row = '1', column = '0')

#далее работаем над окном для основной задачи
lbl = Label(tab1, text="Введите максимальное количество итераций N_max:", font=(18))
lbl.grid(row = '2', column = '0', sticky = E)
txt_N_max = Entry(tab1, width=20)
txt_N_max.grid(row = '2', column = '1', sticky = W)
txt_N_max.insert(0,'10000')

lbl = Label(tab1, text="Введите точность вычислений eps:", font=(18))
lbl.grid(row = '3', column = '0', sticky = E)
txt_eps = Entry(tab1, width=20)
txt_eps.grid(row = '3', column = '1', sticky = W)
txt_eps.insert(0,'0.000001')

lbl = Label(tab1, text="Границы заданы условием, но их можно поменять", font=(18))
lbl.grid(row = '4', column = '0', sticky = E)

lbl = Label(tab1, text="a =", font=(18))
lbl.grid(row = '5', column = '0', sticky = E)
txt_a = Entry(tab1, width=20)
txt_a.grid(row = '5', column = '1', sticky = W)
txt_a.insert(0,'1')

lbl = Label(tab1, text="b =", font=(18))
lbl.grid(row = '6', column = '0', sticky = E)
txt_b = Entry(tab1, width=20)
txt_b.grid(row = '6', column = '1', sticky = W)
txt_b.insert(0,'2')

lbl = Label(tab1, text="c =", font=(18))
lbl.grid(row = '7', column = '0', sticky = E)
txt_c = Entry(tab1, width=20)
txt_c.grid(row = '7', column = '1', sticky = W)
txt_c.insert(0,'2')

lbl = Label(tab1, text="d =", font=(18))
lbl.grid(row = '8', column = '0', sticky = E)
txt_d = Entry(tab1, width=20)
txt_d.grid(row = '8', column = '1', sticky = W)
txt_d.insert(0,'3')

lbl = Label(tab1, text="Выберите метод решения: ", font=(18))
lbl.grid(row = '9', column = '0', sticky = W)

Var_1=IntVar()
Var_1.set(1)
Cheb=Radiobutton(tab1, text='Метод Чебышева',variable=Var_1, value=1, font=(18))
Cheb.grid(row = '10', column = '0', sticky = W)
Simple=Radiobutton(tab1, text='Метод простых итераций', variable=Var_1, value = 2, font=(18))
Simple.grid(row = '11', column = '0', sticky = W)

lbl = Label(tab1, text="Для метода Чебышева укажите число параметров (k):", font=(18))
lbl.grid(row = '12', column = '0', sticky = W)
txt_k = Entry(tab1, width=20)
txt_k.grid(row = '12', column = '1', sticky = W)
txt_k.insert(0,'5')

lbl = Label(tab1, text="Укажите n:     ", font=(18))
lbl.grid(row = '13', column = '0', sticky = E)
txt_n = Entry(tab1, width=20)
txt_n.grid(row = '13', column = '1', sticky = W)
txt_n.insert(0,'20')

lbl = Label(tab1, text="Укажите m:     ", font=(18))
lbl.grid(row = '14', column = '0', sticky = E)
txt_m = Entry(tab1, width=20)
txt_m.grid(row = '14', column = '1', sticky = W)
txt_m.insert(0,'20')

btn_exit = Button(window, text="Выход", command=exit)
btn_exit.place(x='700', y='720')

#окно для тестовой задачи
lbl = Label(tab2, text="Введите максимальное количество итераций N_max:", font=(18))
lbl.grid(row = '2', column = '0', sticky = E)
txt_N_max_2 = Entry(tab2, width=20)
txt_N_max_2.grid(row = '2', column = '1', sticky = W)
txt_N_max_2.insert(0,'10000')

lbl = Label(tab2, text="Введите точность вычислений eps:", font=(18))
lbl.grid(row = '3', column = '0', sticky = E)
txt_eps_2 = Entry(tab2, width=20)
txt_eps_2.grid(row = '3', column = '1', sticky = W)
txt_eps_2.insert(0,'0.000001')

lbl = Label(tab2, text="Границы заданы условием, но их можно поменять", font=(18))
lbl.grid(row = '4', column = '0', sticky = E)

lbl = Label(tab2, text="a =", font=(18))
lbl.grid(row = '5', column = '0', sticky = E)
txt_a_2 = Entry(tab2, width=20)
txt_a_2.grid(row = '5', column = '1', sticky = W)
txt_a_2.insert(0,'1')

lbl = Label(tab2, text="b =", font=(18))
lbl.grid(row = '6', column = '0', sticky = E)
txt_b_2 = Entry(tab2, width=20)
txt_b_2.grid(row = '6', column = '1', sticky = W)
txt_b_2.insert(0,'2')

lbl = Label(tab2, text="c =", font=(18))
lbl.grid(row = '7', column = '0', sticky = E)
txt_c_2 = Entry(tab2, width=20)
txt_c_2.grid(row = '7', column = '1', sticky = W)
txt_c_2.insert(0,'2')

lbl = Label(tab2, text="d =", font=(18))
lbl.grid(row = '8', column = '0', sticky = E)
txt_d_2 = Entry(tab2, width=20)
txt_d_2.grid(row = '8', column = '1', sticky = W)
txt_d_2.insert(0,'3')

lbl = Label(tab2, text="Выберите метод решения: ", font=(18))
lbl.grid(row = '9', column = '0', sticky = W)

Var_2=IntVar()
Var_2.set(1)

Cheb=Radiobutton(tab2, text='Метод Чебышева',variable=Var_2, value=1, font=(18))
Cheb.grid(row = '10', column = '0', sticky = W)

Simple=Radiobutton(tab2, text='Метод простых итераций', variable=Var_2, value = 2, font=(18))
Simple.grid(row = '11', column = '0', sticky = W)

lbl = Label(tab2, text="Для метода Чебышева укажите число параметров(k):", font=(18))
lbl.grid(row = '12', column = '0', sticky = W)
txt_k_2 = Entry(tab2, width=20)
txt_k_2.grid(row = '12', column = '1', sticky = W)
txt_k_2.insert(0,'5')


lbl = Label(tab2, text="Укажите n:     ", font=(18))
lbl.grid(row = '13', column = '0', sticky = E)
txt_n_2 = Entry(tab2, width=20)
txt_n_2.grid(row = '13', column = '1', sticky = W)
txt_n_2.insert(0,'20')

lbl = Label(tab2, text="Укажите m:     ", font=(18))
lbl.grid(row = '14', column = '0', sticky = E)
txt_m_2 = Entry(tab2, width=20)
txt_m_2.grid(row = '14', column = '1', sticky = W)
txt_m_2.insert(0,'20')

###
tab_control.pack(expand=1, fill='both') 
window.mainloop()