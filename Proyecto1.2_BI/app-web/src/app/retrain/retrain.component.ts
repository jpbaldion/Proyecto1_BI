import { Component } from '@angular/core';
import { AppService } from '../app.service';
import Papa from 'papaparse';  // Importar PapaParse

@Component({
  selector: 'app-retrain',
  templateUrl: './retrain.component.html',
  styleUrls: ['./retrain.component.css']
})
export class RetrainComponent {
  csvFile: File | null = null;
  statusMessage: string = '';
  retrainDisabled: boolean = true;
  metricas: any = null;  // Propiedad para almacenar las métricas

  constructor(private appService: AppService) {}

  // Manejar la carga del archivo CSV usando PapaParse
  onFileChange(event: any): void {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.csv')) {
      this.csvFile = file;
      this.retrainDisabled = false;
      this.statusMessage = 'Archivo cargado correctamente.';
    } else {
      this.statusMessage = 'Por favor, selecciona un archivo CSV.';
      this.retrainDisabled = true;
    }
  }

  // Función para leer y analizar el CSV usando PapaParse
  parseCSV(file: File): Promise<any> {
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        complete: (result) => resolve(result.data),
        error: (error) => reject(error),
        header: true,   // Considerar la primera fila como cabecera
        skipEmptyLines: true, // Omitir líneas vacías
        delimiter: ';'  // Establecer delimitador adecuado
      });
    });
  }

  // Enviar el archivo CSV para reentrenar el modelo
  onRetrainClick(): void {
    if (!this.csvFile) {
      this.statusMessage = 'Por favor, selecciona un archivo CSV primero.';
      return;
    }

    // Leer y procesar el archivo CSV usando PapaParse
    this.parseCSV(this.csvFile).then((data) => {
      // Crear el objeto JSON que el backend espera
      const noticias = data.map((row: any) => ({
        titulo: row.Titulo,
        descripcion: row.Descripcion,
        label: parseInt(row.Label, 10)
      }));

      // Llamar al servicio para reentrenar el modelo
      this.appService.reentrenarModelo(noticias).subscribe(
        (response) => {
          this.statusMessage = response.mensaje;  // Mostrar el mensaje de éxito
          this.metricas = response.metricas;  // Almacenar las métricas en la propiedad
        },
        (error) => {
          this.statusMessage = `Error: ${error.message}`;
        }
      );
    }).catch((error) => {
      this.statusMessage = `Error al procesar el archivo CSV: ${error.message}`;
    });
  }
}
